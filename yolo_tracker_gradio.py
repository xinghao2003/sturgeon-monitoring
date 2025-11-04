import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_TRACKER = "botsort.yaml"

CV_COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "HOT": cv2.COLORMAP_HOT,
    "BONE": cv2.COLORMAP_BONE,
    "VIRIDIS": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
}


def _resolve_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path")
    return getattr(file_obj, "name", None)


def _parse_class_filter(raw: str) -> Optional[List[int]]:
    if not raw:
        return None
    tokens = raw.replace(";", ",").split(",")
    out: List[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            pass
    return out or None


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _class_label(names: Iterable, cid: int) -> str:
    if isinstance(names, dict):
        return f"{cid}-{names.get(cid, cid)}"
    if isinstance(names, list) and 0 <= cid < len(names):
        return f"{cid}-{names[cid]}"
    return str(cid)


def _splat_gaussian(acc: np.ndarray, cx: int, cy: int, sigma: int) -> None:
    if sigma <= 0:
        return
    h, w = acc.shape[:2]
    r = int(3 * sigma)
    x0 = max(cx - r, 0)
    x1 = min(cx + r + 1, w)
    y0 = max(cy - r, 0)
    y1 = min(cy + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma)).astype(np.float32)
    total = g.sum()
    if total > 1e-6:
        g /= total
    acc[y0:y1, x0:x1] += g


def _heatmap_rgb(acc: np.ndarray, percentile: int, cmap_key: str) -> np.ndarray:
    if acc is None or acc.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    percentile = max(90, min(100, int(percentile)))
    scale = np.percentile(acc, percentile)
    denom = max(scale, 1e-6)
    norm = np.clip(acc / denom, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    cmap = CV_COLORMAPS.get(cmap_key, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _init_writer(path: Path, size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {path}")
    return writer


def _maybe_zip_per_class(per_class_dir: Path) -> Optional[Path]:
    if not per_class_dir.exists():
        return None
    if not any(per_class_dir.iterdir()):
        return None
    archive_base = per_class_dir.parent / f"{per_class_dir.name}"
    archive_path = shutil.make_archive(str(archive_base), "zip", per_class_dir)
    return Path(archive_path)


def _create_chunk_writer(path: Path, size: Tuple[int, int], fps: float, chunk_frames: int = 30):
    """Create a writer that yields video chunks as they're written."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    chunk_path = path.parent / f"{path.stem}_chunk_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(str(chunk_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {chunk_path}")
    return writer, chunk_path


def run_tracking(
    model_file,
    video_file,
    device: str,
    tracker_cfg: str,
    conf: float,
    iou: float,
    max_det: int,
    class_filter: str,
    show_ids: bool,
    heat_overlay: bool,
    heat_opacity: float,
    heat_export: bool,
    per_class_export: bool,
    sigma_factor: float,
    percentile: int,
    decay_enabled: bool,
    decay_rate: int,
    colormap: str,
    source_mode: str,
    camera_source: str,
    live_duration: int,
    simulate_real_time: bool,
    enable_streaming: bool = False,
):
    progress = gr.Progress(track_tqdm=False)
    status: List[str] = []
    progress(0, desc="Preparing inputs")

    model_path = _resolve_path(model_file)
    video_path = _resolve_path(video_file)

    if not model_path or not Path(model_path).exists():
        return None, None, None, "Model file not provided or missing."

    source_mode = (source_mode or "uploaded").lower()
    camera_source = (camera_source or "0").strip()
    live_duration = max(0, int(live_duration or 0))
    if source_mode not in {"uploaded", "demo", "live"}:
        return None, None, None, f"Unsupported source mode: {source_mode}"
    if source_mode in {"uploaded", "demo"}:
        if not video_path or not Path(video_path).exists():
            return None, None, None, "Video file not provided or missing."
        if source_mode == "demo":
            simulate_real_time = True

    class_ids = _parse_class_filter(class_filter)

    sigma_factor = max(0.10, min(1.0, float(sigma_factor)))
    heat_opacity = max(0.0, min(1.0, float(heat_opacity)))
    decay = max(0.0, min(0.95, float(decay_rate) / 100.0 if decay_enabled else 0.0))
    max_det = max(1, int(max_det))

    progress(0, desc="Loading model")
    try:
        model = YOLO(model_path, task="segment")
    except Exception as exc:
        return None, None, None, f"Failed to load model: {exc}"

    model_names = getattr(model, "names", {})
    applied_device = device
    try:
        if device != "auto":
            model.to(device)
    except Exception as exc:
        status.append(f"Device '{device}' unavailable ({exc}); using auto.")
        applied_device = "auto"

    if source_mode == "live":
        try:
            capture_source = int(camera_source)
        except ValueError:
            capture_source = camera_source
        cap = cv2.VideoCapture(capture_source)
    else:
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None, "Unable to open video source."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    live_frame_cap: Optional[int] = None
    if source_mode == "live" and live_duration > 0 and fps > 0:
        live_frame_cap = int(fps * live_duration)
        total_frames = live_frame_cap

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preload_frame: Optional[np.ndarray] = None
    if width <= 0 or height <= 0:
        ok_probe, probe_frame = cap.read()
        if not ok_probe or probe_frame is None:
            cap.release()
            return None, None, None, "Unable to determine video geometry."
        height, width = probe_frame.shape[:2]
        if source_mode == "live":
            preload_frame = probe_frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    size = (width, height)

    temp_dir = Path(tempfile.mkdtemp(prefix="sturgeon_tracker_"))
    stem = Path(video_path).stem if video_path else f"{source_mode}_source"
    tracked_path = temp_dir / f"{stem}_tracked.mp4"
    heatmap_path = temp_dir / f"{stem}_heatmap.mp4"
    per_class_dir = temp_dir / "per_class"
    per_class_dir.mkdir(exist_ok=True)

    annotated_writer = _init_writer(tracked_path, size, fps)
    heat_writer = _init_writer(heatmap_path, size, fps) if heat_export else None
    per_class_writers: Dict[int, cv2.VideoWriter] = {}

    heat_accum: Optional[np.ndarray] = None
    heat_accum_per_cls: Dict[int, np.ndarray] = {}
    processed = 0
    frames_in_window = 0
    window_start = time.time()

    try:
        progress(0, desc="Running tracking")
        while True:
            frame_start = time.time()
            if preload_frame is not None:
                frame_bgr = preload_frame
                preload_frame = None
                ok = True
            else:
                ok, frame_bgr = cap.read()
            if not ok:
                break
            processed += 1

            results = model.track(
                source=frame_bgr,
                persist=True,
                tracker=tracker_cfg or DEFAULT_TRACKER,
                conf=float(conf),
                iou=float(iou),
                max_det=max_det,
                classes=class_ids,
                verbose=False,
            )
            res = results[0]
            boxes = getattr(res, "boxes", None)

            xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and getattr(boxes, "xyxy", None) is not None else None
            cls_arr = boxes.cls.int().cpu().numpy() if boxes is not None and getattr(boxes, "cls", None) is not None else None

            if heat_accum is None:
                heat_accum = np.zeros((height, width), dtype=np.float32)
            if per_class_export:
                for cid in (class_ids or getattr(model, "names", {})):
                    if cid not in heat_accum_per_cls:
                        heat_accum_per_cls[cid] = np.zeros((height, width), dtype=np.float32)

            if decay > 0 and heat_accum is not None:
                heat_accum *= (1.0 - decay)
                for cid in list(heat_accum_per_cls.keys()):
                    heat_accum_per_cls[cid] *= (1.0 - decay)

            if xyxy is not None and heat_accum is not None:
                for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                    cx = int(0.5 * (x1 + x2))
                    cy = int(0.5 * (y1 + y2))
                    bw = max(int(x2 - x1), 1)
                    bh = max(int(y2 - y1), 1)
                    sigma = max(3, int(min(bw, bh) * sigma_factor))
                    _splat_gaussian(heat_accum, cx, cy, sigma)
                    if per_class_export and cls_arr is not None:
                        cid = int(cls_arr[idx])
                        if cid not in heat_accum_per_cls:
                            heat_accum_per_cls[cid] = np.zeros((height, width), dtype=np.float32)
                        _splat_gaussian(heat_accum_per_cls[cid], cx, cy, sigma)

            annotated_bgr = res.plot()

            if show_ids and boxes is not None and getattr(boxes, "id", None) is not None:
                ids = boxes.id.int().cpu().tolist()
                bxy = boxes.xyxy.int().cpu().tolist()
                for (x1, y1, x2, y2), tid in zip(bxy, ids):
                    cv2.putText(
                        annotated_bgr,
                        f"ID {tid}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            heat_rgb = _heatmap_rgb(heat_accum, percentile, colormap) if heat_accum is not None else None
            if heat_overlay and heat_rgb is not None:
                annotated_bgr = cv2.addWeighted(
                    annotated_bgr,
                    1.0,
                    cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR),
                    heat_opacity,
                    0,
                )

            annotated_writer.write(annotated_bgr)
            if heat_writer is not None and heat_rgb is not None:
                heat_writer.write(cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

            if per_class_export and cls_arr is not None and heat_accum_per_cls:
                for cid, acc in heat_accum_per_cls.items():
                    if cid not in per_class_writers:
                        per_class_path = per_class_dir / f"{stem}_heatmap_cls-{_class_label(model_names, cid)}.mp4"
                        per_class_writers[cid] = _init_writer(per_class_path, size, fps)
                    heat_rgb_cls = _heatmap_rgb(acc, percentile, colormap)
                    per_class_writers[cid].write(cv2.cvtColor(heat_rgb_cls, cv2.COLOR_RGB2BGR))

            frames_in_window += 1
            now = time.time()
            if now - window_start >= 0.5:
                fps_live = frames_in_window / (now - window_start)
                window_start = now
                frames_in_window = 0
                status_line = f"Processed {processed}/{total_frames or '?'} frames | live FPS ≈ {fps_live:.1f}"
                status.append(status_line)
                if len(status) > 6:
                    status = status[-6:]

            if total_frames > 0:
                progress(min(processed / total_frames, 0.999), desc=f"Frames: {processed}/{total_frames}")
            else:
                progress(desc=f"Frames processed: {processed}")

            if live_frame_cap is not None and processed >= live_frame_cap:
                status.append(f"Reached live capture limit: {live_duration} seconds")
                break

            if simulate_real_time and fps > 1e-3:
                target_dt = 1.0 / fps
                elapsed = time.time() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

    except Exception as exc:
        return None, None, None, f"Processing error: {exc}"
    finally:
        cap.release()
        annotated_writer.release()
        if heat_writer is not None:
            heat_writer.release()
        for writer in per_class_writers.values():
            writer.release()

    per_class_zip = _maybe_zip_per_class(per_class_dir) if per_class_export else None
    summary = [
        f"Device: {applied_device}",
        f"Source mode: {source_mode}",
        f"Frames processed: {processed}",
        f"Duration: {_fmt_time(processed / fps) if processed else '0:00'}",
        f"Tracked video: {tracked_path}",
    ]
    if source_mode == "live":
        summary.append(f"Capture target: {camera_source or '0'}")
        summary.append(f"Live duration target: {live_duration} s")
    if simulate_real_time:
        summary.append("Playback throttled to source FPS.")
    if heat_export:
        summary.append(f"Heatmap video: {heatmap_path}")
    if per_class_zip:
        summary.append(f"Per-class archive: {per_class_zip}")
    summary.extend(status[-4:])

    progress(1.0, desc="Done")
    return str(tracked_path), (str(heatmap_path) if heat_export else None), (str(per_class_zip) if per_class_zip else None), "\n".join(summary)


def run_tracking_stream(
    model_file,
    video_file,
    device: str,
    tracker_cfg: str,
    conf: float,
    iou: float,
    max_det: int,
    class_filter: str,
    show_ids: bool,
    heat_overlay: bool,
    heat_opacity: float,
    heat_export: bool,
    per_class_export: bool,
    sigma_factor: float,
    percentile: int,
    decay_enabled: bool,
    decay_rate: int,
    colormap: str,
    source_mode: str,
    camera_source: str,
    live_duration: int,
    simulate_real_time: bool,
):
    """Generator version of tracking that yields video chunks for streaming."""
    status_updates: List[str] = []
    model_path = _resolve_path(model_file)
    video_path = _resolve_path(video_file)

    if not model_path or not Path(model_path).exists():
        yield None, None, None, "Model file not provided or missing."
        return

    source_mode = (source_mode or "uploaded").lower()
    camera_source = (camera_source or "0").strip()
    live_duration = max(0, int(live_duration or 0))
    if source_mode not in {"uploaded", "demo", "live"}:
        yield None, None, None, f"Unsupported source mode: {source_mode}"
        return
    if source_mode in {"uploaded", "demo"}:
        if not video_path or not Path(video_path).exists():
            yield None, None, None, "Video file not provided or missing."
            return
        if source_mode == "demo":
            simulate_real_time = True

    class_ids = _parse_class_filter(class_filter)

    sigma_factor = max(0.10, min(1.0, float(sigma_factor)))
    heat_opacity = max(0.0, min(1.0, float(heat_opacity)))
    decay = max(0.0, min(0.95, float(decay_rate) / 100.0 if decay_enabled else 0.0))
    max_det = max(1, int(max_det))

    try:
        model = YOLO(model_path, task="segment")
    except Exception as exc:
        yield None, None, None, f"Failed to load model: {exc}"
        return

    model_names = getattr(model, "names", {})
    applied_device = device
    try:
        if device != "auto":
            model.to(device)
    except Exception as exc:
        status.append(f"Device '{device}' unavailable ({exc}); using auto.")
        applied_device = "auto"

    if source_mode == "live":
        try:
            capture_source = int(camera_source)
        except ValueError:
            capture_source = camera_source
        cap = cv2.VideoCapture(capture_source)
    else:
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        yield None, None, None, "Unable to open video source."
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    live_frame_cap: Optional[int] = None
    if source_mode == "live" and live_duration > 0 and fps > 0:
        live_frame_cap = int(fps * live_duration)
        total_frames = live_frame_cap

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preload_frame: Optional[np.ndarray] = None
    if width <= 0 or height <= 0:
        ok_probe, probe_frame = cap.read()
        if not ok_probe or probe_frame is None:
            cap.release()
            yield None, None, None, "Unable to determine video geometry."
            return
        height, width = probe_frame.shape[:2]
        if source_mode == "live":
            preload_frame = probe_frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    size = (width, height)

    temp_dir = Path(tempfile.mkdtemp(prefix="sturgeon_tracker_"))
    stem = Path(video_path).stem if video_path else f"{source_mode}_source"
    tracked_path = temp_dir / f"{stem}_tracked.mp4"
    heatmap_path = temp_dir / f"{stem}_heatmap.mp4"
    per_class_dir = temp_dir / "per_class"
    per_class_dir.mkdir(exist_ok=True)

    annotated_writer = _init_writer(tracked_path, size, fps)
    heat_writer = _init_writer(heatmap_path, size, fps) if heat_export else None
    per_class_writers: Dict[int, cv2.VideoWriter] = {}

    heat_accum: Optional[np.ndarray] = None
    heat_accum_per_cls: Dict[int, np.ndarray] = {}
    processed = 0
    frames_in_window = 0
    window_start = time.time()
    chunk_frames = max(30, int(fps * 1.0))  # ~1 second worth of frames per chunk
    frames_since_chunk = 0

    try:
        while True:
            frame_start = time.time()
            if preload_frame is not None:
                frame_bgr = preload_frame
                preload_frame = None
                ok = True
            else:
                ok, frame_bgr = cap.read()
            if not ok:
                break
            processed += 1

            results = model.track(
                source=frame_bgr,
                persist=True,
                tracker=tracker_cfg or DEFAULT_TRACKER,
                conf=float(conf),
                iou=float(iou),
                max_det=max_det,
                classes=class_ids,
                verbose=False,
            )
            res = results[0]
            boxes = getattr(res, "boxes", None)

            xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and getattr(boxes, "xyxy", None) is not None else None
            cls_arr = boxes.cls.int().cpu().numpy() if boxes is not None and getattr(boxes, "cls", None) is not None else None

            if heat_accum is None:
                heat_accum = np.zeros((height, width), dtype=np.float32)
            if per_class_export:
                for cid in (class_ids or getattr(model, "names", {})):
                    if cid not in heat_accum_per_cls:
                        heat_accum_per_cls[cid] = np.zeros((height, width), dtype=np.float32)

            if decay > 0 and heat_accum is not None:
                heat_accum *= (1.0 - decay)
                for cid in list(heat_accum_per_cls.keys()):
                    heat_accum_per_cls[cid] *= (1.0 - decay)

            if xyxy is not None and heat_accum is not None:
                for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                    cx = int(0.5 * (x1 + x2))
                    cy = int(0.5 * (y1 + y2))
                    bw = max(int(x2 - x1), 1)
                    bh = max(int(y2 - y1), 1)
                    sigma = max(3, int(min(bw, bh) * sigma_factor))
                    _splat_gaussian(heat_accum, cx, cy, sigma)
                    if per_class_export and cls_arr is not None:
                        cid = int(cls_arr[idx])
                        if cid not in heat_accum_per_cls:
                            heat_accum_per_cls[cid] = np.zeros((height, width), dtype=np.float32)
                        _splat_gaussian(heat_accum_per_cls[cid], cx, cy, sigma)

            annotated_bgr = res.plot()

            if show_ids and boxes is not None and getattr(boxes, "id", None) is not None:
                ids = boxes.id.int().cpu().tolist()
                bxy = boxes.xyxy.int().cpu().tolist()
                for (x1, y1, x2, y2), tid in zip(bxy, ids):
                    cv2.putText(
                        annotated_bgr,
                        f"ID {tid}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            heat_rgb = _heatmap_rgb(heat_accum, percentile, colormap) if heat_accum is not None else None
            if heat_overlay and heat_rgb is not None:
                annotated_bgr = cv2.addWeighted(
                    annotated_bgr,
                    1.0,
                    cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR),
                    heat_opacity,
                    0,
                )

            annotated_writer.write(annotated_bgr)
            if heat_writer is not None and heat_rgb is not None:
                heat_writer.write(cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

            if per_class_export and cls_arr is not None and heat_accum_per_cls:
                for cid, acc in heat_accum_per_cls.items():
                    if cid not in per_class_writers:
                        per_class_path = per_class_dir / f"{stem}_heatmap_cls-{_class_label(model_names, cid)}.mp4"
                        per_class_writers[cid] = _init_writer(per_class_path, size, fps)
                    heat_rgb_cls = _heatmap_rgb(acc, percentile, colormap)
                    per_class_writers[cid].write(cv2.cvtColor(heat_rgb_cls, cv2.COLOR_RGB2BGR))

            frames_in_window += 1
            frames_since_chunk += 1
            now = time.time()
            if now - window_start >= 0.5:
                fps_live = frames_in_window / (now - window_start)
                window_start = now
                frames_in_window = 0
                status_line = f"Processed {processed}/{total_frames or '?'} frames | live FPS ≈ {fps_live:.1f}"
                status_updates.append(status_line)
                if len(status_updates) > 6:
                    status_updates = status_updates[-6:]

            # Yield chunk every N frames
            if frames_since_chunk >= chunk_frames:
                annotated_writer.release()
                yield str(tracked_path), None, None, f"Processing: {processed} frames"
                annotated_writer = _init_writer(tracked_path, size, fps)
                frames_since_chunk = 0

            if live_frame_cap is not None and processed >= live_frame_cap:
                status_updates.append(f"Reached live capture limit: {live_duration} seconds")
                break

            if simulate_real_time and fps > 1e-3:
                target_dt = 1.0 / fps
                elapsed = time.time() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

    except Exception as exc:
        yield None, None, None, f"Processing error: {exc}"
        return
    finally:
        cap.release()
        annotated_writer.release()
        if heat_writer is not None:
            heat_writer.release()
        for writer in per_class_writers.values():
            writer.release()

    per_class_zip = _maybe_zip_per_class(per_class_dir) if per_class_export else None
    summary = [
        f"Device: {applied_device}",
        f"Source mode: {source_mode}",
        f"Frames processed: {processed}",
        f"Duration: {_fmt_time(processed / fps) if processed else '0:00'}",
        f"Tracked video: {tracked_path}",
    ]
    if source_mode == "live":
        summary.append(f"Capture target: {camera_source or '0'}")
        summary.append(f"Live duration target: {live_duration} s")
    if simulate_real_time:
        summary.append("Playback throttled to source FPS.")
    if heat_export:
        summary.append(f"Heatmap video: {heatmap_path}")
    if per_class_zip:
        summary.append(f"Per-class archive: {per_class_zip}")
    summary.extend(status_updates[-4:])

    yield str(tracked_path), (str(heatmap_path) if heat_export else None), (str(per_class_zip) if per_class_zip else None), "\n".join(summary)


with gr.Blocks(title="YOLOv11 Tracker") as demo:
    gr.Markdown("## YOLOv11 Tracker (Gradio)")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            video_file = gr.Video(
                label="Video / Stream",
                sources=["upload", "webcam"],
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            with gr.Row():
                run_btn = gr.Button("Run tracking", variant="primary")
                stream_btn = gr.Button("Run tracking (Stream)", variant="secondary")
            
            tracked_out = gr.Video(label="Tracked video (Live)", streaming=True, autoplay=True)
            tracked_out_file = gr.File(label="Tracked video (Download)")
            
            with gr.Row():
                heatmap_out = gr.File(label="Heatmap video")
                per_class_out = gr.File(label="Per-class heatmaps ZIP")
            log = gr.Textbox(label="Run summary", lines=10)

        with gr.Column(scale=2):
            with gr.Accordion("Model & Source", open=True):
                model_file = gr.File(label="YOLO model (.pt / .onnx)")
                source_mode = gr.Dropdown(
                    ["uploaded", "demo", "live"],
                    value="uploaded",
                    label="Input source",
                    info="Demo throttles playback, live uses OpenCV camera index or stream URL.",
                )
                camera_source = gr.Textbox(
                    value="0",
                    label="Camera index / stream URL",
                    info="Used when source is live.",
                )
                live_duration = gr.Slider(
                    5,
                    600,
                    value=120,
                    step=5,
                    label="Live capture duration (s)",
                )
                simulate_real_time = gr.Checkbox(
                    False,
                    label="Throttle to source FPS",
                    info="Applies to uploaded/demo sources.",
                )
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")

            with gr.Accordion("Tracking", open=True):
                tracker = gr.Dropdown(["botsort.yaml", "bytetrack.yaml"], value=DEFAULT_TRACKER, label="Tracker")
                class_filter = gr.Textbox(
                    label="Class filter (comma-separated IDs)",
                    placeholder="Leave empty for all",
                )
                conf = gr.Slider(0.05, 1.0, value=DEFAULT_CONF, step=0.01, label="Confidence")
                iou = gr.Slider(0.05, 1.0, value=DEFAULT_IOU, step=0.01, label="IoU")
                max_det = gr.Slider(20, 300, value=120, step=10, label="Max detections")
                show_ids = gr.Checkbox(True, label="Show track IDs")

            with gr.Accordion("Heatmap", open=False):
                heat_overlay = gr.Checkbox(True, label="Overlay heatmap")
                heat_opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Heat overlay opacity")
                colormap = gr.Dropdown(list(CV_COLORMAPS.keys()), value="JET", label="Colormap")
                heat_export = gr.Checkbox(False, label="Export heatmap video")
                per_class_export = gr.Checkbox(False, label="Per-class heatmaps (ZIP)")
                sigma_factor = gr.Slider(0.10, 1.00, value=0.25, step=0.05, label="Sigma factor × box size")
                percentile = gr.Slider(90, 100, value=99, step=1, label="Percentile cap")
                decay_enabled = gr.Checkbox(False, label="Temporal decay")
                decay_rate = gr.Slider(0, 20, value=0, step=1, label="Decay per frame (%)")

    run_btn.click(
        run_tracking,
        inputs=[
            model_file,
            video_file,
            device,
            tracker,
            conf,
            iou,
            max_det,
            class_filter,
            show_ids,
            heat_overlay,
            heat_opacity,
            heat_export,
            per_class_export,
            sigma_factor,
            percentile,
            decay_enabled,
            decay_rate,
            colormap,
            source_mode,
            camera_source,
            live_duration,
            simulate_real_time,
        ],
        outputs=[tracked_out_file, heatmap_out, per_class_out, log],
    )

    stream_btn.click(
        run_tracking_stream,
        inputs=[
            model_file,
            video_file,
            device,
            tracker,
            conf,
            iou,
            max_det,
            class_filter,
            show_ids,
            heat_overlay,
            heat_opacity,
            heat_export,
            per_class_export,
            sigma_factor,
            percentile,
            decay_enabled,
            decay_rate,
            colormap,
            source_mode,
            camera_source,
            live_duration,
            simulate_real_time,
        ],
        outputs=[tracked_out, heatmap_out, per_class_out, log],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
