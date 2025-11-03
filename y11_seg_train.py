# y11_seg_train.py
# Examples:
#   python y11_seg_train.py --data data.yaml --model yolo11s-seg.pt --epochs 100 --export-onnx
#   python y11_seg_train.py --data data.yaml --tune --tune-iterations 200 --tune-epochs 30 --export-onnx
#   python y11_seg_train.py --data data.yaml --resume runs/segment/train5/weights/last.pt --export-onnx --export-opset 13

import argparse, shutil, json, yaml, os, csv, hashlib, sys, platform, time
from pathlib import Path
from datetime import datetime
from rich import print

# TorchVision's CUDA NMS bindings are frequently missing on Windows wheels; fall back to CPU NMS.
if platform.system().lower().startswith("win") and "TORCHVISION_DISABLE_NMS_CUDA" not in os.environ:
    os.environ["TORCHVISION_DISABLE_NMS_CUDA"] = "1"

from ultralytics import YOLO

# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser("YOLOv11 Instance Segmentation: tune/resume/val/test + ONNX export + rich artifacts")
    default_workers = 0 if platform.system().lower().startswith("win") else None
    # core
    p.add_argument("--data", required=True, help="Ultralytics data YAML")
    p.add_argument("--model", default="yolo11n-seg.pt", help="Base model (yolo11[n/s/m/l/x]-seg.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", default=None, help="int or -1 for auto")
    p.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu or mps")
    p.add_argument("--project", default="runs_seg")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None, help="path/to/last.pt to resume training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--workers", type=int, default=default_workers,
                   help="DataLoader workers (defaults to 0 on Windows; otherwise Ultralytics default)")

    # tuning
    p.add_argument("--tune", action="store_true")
    p.add_argument("--tune-iterations", type=int, default=300)
    p.add_argument("--tune-epochs", type=int, default=30)
    p.add_argument("--tune-name", default=None)
    p.add_argument("--tune-workers", type=int, default=None,
                   help="Override DataLoader workers specifically during hyperparameter tuning")
    p.add_argument("--tuned-hyp", default=None, help="Path to best_hyperparameters.yaml to apply for training")

    # export
    p.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX")
    p.add_argument("--export-opset", type=int, default=13, help="ONNX opset")
    p.add_argument("--export-dynamic", action="store_true", help="Enable dynamic axes in ONNX")
    p.add_argument("--export-simplify", action="store_true", help="Simplify ONNX graph if supported")

    return p.parse_args()

# ------------------------ Utils ------------------------

def pathify(x):
    return None if x is None else Path(x)

def copy_if_exists(src, dst_dir):
    src = Path(src)
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        return (dst_dir / src.name).as_posix()
    return None

def sha256sum(p: Path, chunk=1024 * 1024):
    if not p or not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def read_results_csv(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # convert numeric strings to float where possible
            for k, v in row.items():
                if v is None: continue
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows

def summarize_training(rows):
    """Return a tiny summary (best epoch by mAP50-95 if present; else by metrics/mAP50 or F1)."""
    if not rows:
        return {}
    # Heuristics for common Ultralytics columns
    key_order = ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "metrics/mAP50(B)", "metrics/mAP50", "mAP50", "metrics/F1(B)", "metrics/F1"]
    best_key = next((k for k in key_order if k in rows[-1]), None)
    if best_key is None:
        return {"best_key": None, "note": "No known metric column found."}

    best_idx = max(range(len(rows)), key=lambda i: rows[i].get(best_key, float("-inf")))
    best_row = rows[best_idx]
    return {
        "metric": best_key,
        "best_epoch": int(best_row.get("epoch", best_idx)),
        "best_value": float(best_row.get(best_key, float("nan"))),
        "last_epoch": int(rows[-1].get("epoch", len(rows) - 1)),
        "last_value": float(rows[-1].get(best_key, float("nan"))),
    }

def collect_run_artifacts(run_dir: Path, artifacts_dir: Path, tag: str):
    out = {}
    dst = artifacts_dir / tag
    dst.mkdir(parents=True, exist_ok=True)

    for fname in [
        "results.csv", "results.png",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
        "labels_correlogram.jpg", "val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg",
        "args.yaml", "hyp.yaml", "train.log",
    ]:
        p = run_dir / fname
        copied = copy_if_exists(p, dst)
        if copied:
            out[fname] = copied

    # weights
    weights = run_dir / "weights"
    if weights.exists():
        for w in ["best.pt", "last.pt"]:
            copied = copy_if_exists(weights / w, dst)
            if copied: out[w] = copied

    with open(dst / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

def write_events_jsonl(rows, out_path: Path):
    if not rows:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path.as_posix()

def system_info_dict():
    try:
        import torch, ultralytics
        device_cnt = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(device_cnt)] if device_cnt else []
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ultralytics": getattr(__import__("ultralytics"), "__version__", "unknown"),
            "torch": getattr(__import__("torch"), "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "num_gpus": device_cnt,
            "gpus": gpu_names,
        }
    except Exception as e:
        return {"error_collecting_system_info": str(e)}

# ------------------------ Main ------------------------

def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    project = args.project
    run_name = args.name or f"y11seg_{ts}"

    artifacts_dir = Path(project) / (run_name + "_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build/Load model
    if args.resume:
        print(f"[bold yellow]Resuming training from[/] {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    # Optional tuning
    best_hyp = None
    tune_run_dir = None
    tune_error = None
    if args.tune:
        print("[bold cyan]Running hyperparameter tuning...[/]")
        search_space = {
            "lr0": (1e-4, 1e-2),
            "degrees": (0.0, 20.0),
            "scale": (0.2, 0.8),
            "hsv_s": (0.3, 0.9),
            "hsv_v": (0.2, 0.8),
            "copy_paste": (0.0, 0.8),
        }

        tune_workers = args.tune_workers if args.tune_workers is not None else args.workers
        tune_kwargs = dict(
            data=args.data,
            epochs=args.tune_epochs,
            iterations=args.tune_iterations,
            space=search_space,
            optimizer="AdamW",
            plots=args.plots,
            save=True,
            val=False,
            device=args.device,
            seed=args.seed,
            project=project,
            name=args.tune_name or (run_name + "_tune"),
            exist_ok=args.exist_ok,
        )
        if tune_workers is not None:
            tune_kwargs["workers"] = tune_workers

        try:
            try:
                model.tune(**tune_kwargs)
            except TypeError:
                resume_kwargs = dict(tune_kwargs)
                resume_kwargs["resume"] = True
                model.tune(**resume_kwargs)
        except Exception as exc:
            tune_error = exc
            print(f"[bold red]Hyperparameter tuning failed:[/] {exc}")

        if tune_error:
            print("[yellow]Continuing without tuned hyperparameters.[/]")
        else:
            # Pull best hyperparameters if produced
            tune_root = Path("runs") / "segment" / "tune"
            candidates = sorted(tune_root.glob("*/best_hyperparameters.yaml"))
            if candidates:
                best_hyp_path = candidates[-1]
                with open(best_hyp_path, "r", encoding="utf-8") as f:
                    best_hyp = yaml.safe_load(f)
                tune_run_dir = best_hyp_path.parent
                shutil.copy2(best_hyp_path, artifacts_dir / "best_hyperparameters.yaml")
                for extra in ["tune_results.csv", "best_fitness.png", "tune_scatter_plots.png"]:
                    copy_if_exists(tune_run_dir / extra, artifacts_dir)
                print(f"[green]Loaded tuned hyperparameters from[/] {best_hyp_path}")
            else:
                print("[yellow]Tuning finished but best_hyperparameters.yaml not found; proceeding with defaults.[/]")
    else:
        print("[bold cyan]Skipping tuning (use --tune to enable).[/]")

    # Training
    print("[bold cyan]Starting training...[/]")
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=project,
        name=run_name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        device=args.device,
        plots=args.plots,
        batch=(None if args.batch is None else (int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch)),
        resume=bool(args.resume),
    )
    if best_hyp:
        train_kwargs.update(best_hyp)
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.tuned_hyp:
        import yaml, pathlib
        hyp_path = pathlib.Path(args.tuned_hyp)
        if hyp_path.exists():
            with open(hyp_path, "r", encoding="utf-8") as f:
                tuned = yaml.safe_load(f) or {}
            # merge tuned hypers into the training kwargs
            train_kwargs.update(tuned)
            print(f"[green]Loaded tuned hypers from[/] {hyp_path}")
        else:
            print(f"[yellow]--tuned-hyp file not found:[/] {hyp_path}")

    train_start = time.time()
    train_results = model.train(**train_kwargs)
    train_end = time.time()
    train_run_dir = Path(train_results.save_dir)  # type: ignore[attr-defined]

    # Validation
    print("[bold cyan]Running validation...[/]")
    val_kwargs = dict(data=args.data, project=project, name=run_name + "_val", plots=args.plots, device=args.device)
    if args.workers is not None:
        val_kwargs["workers"] = args.workers
    val_results = model.val(**val_kwargs)
    val_run_dir = Path(val_results.save_dir)  # type: ignore[attr-defined]

    # Optional test
    try:
        with open(args.data, "r", encoding="utf-8") as f:
            ds = yaml.safe_load(f)
        has_test = "test" in ds and ds["test"]
    except Exception:
        has_test = False

    test_run_dir = None
    if has_test:
        print("[bold cyan]Running test split...[/]")
        test_kwargs = dict(data=args.data, split="test", project=project, name=run_name + "_test",
                           plots=args.plots, device=args.device)
        if args.workers is not None:
            test_kwargs["workers"] = args.workers
        test_results = model.val(**test_kwargs)
        test_run_dir = Path(test_results.save_dir)  # type: ignore[attr-defined]

    # Collect artifacts
    print("[bold cyan]Collecting artifacts...[/]")
    manifest = {
        "project": project,
        "run_name": run_name,
        "model_start": args.model if not args.resume else f"RESUME:{args.resume}",
        "train_run_dir": train_run_dir.as_posix(),
        "val_run_dir": val_run_dir.as_posix(),
        "test_run_dir": test_run_dir.as_posix() if test_run_dir else None,
    }
    manifest["train_files"] = collect_run_artifacts(train_run_dir, artifacts_dir, "train")
    manifest["val_files"] = collect_run_artifacts(val_run_dir, artifacts_dir, "val")
    if test_run_dir:
        manifest["test_files"] = collect_run_artifacts(test_run_dir, artifacts_dir, "test")

    # Training process capture: per-epoch JSONL, summary, env, args snapshot, checksums
    # Per-epoch events (from train results.csv)
    train_csv = Path(manifest["train_files"].get("results.csv") or "")
    rows = read_results_csv(train_csv) if train_csv else []
    summary = summarize_training(rows)

    events_path = artifacts_dir / "train_events.jsonl"
    write_events_jsonl(rows, events_path)

    # Checksums for weights
    best_pt = Path(manifest["train_files"].get("best.pt") or "")
    last_pt = Path(manifest["train_files"].get("last.pt") or "")
    checksums = {
        "best.pt": sha256sum(best_pt) if best_pt else None,
        "last.pt": sha256sum(last_pt) if last_pt else None,
    }

    # System + run info
    proc_info = {
        "system": system_info_dict(),
        "args": vars(args),
        "tuned_hypers": best_hyp or {},
        "tune_error": str(tune_error) if tune_error else None,
        "train_time_sec": round(train_end - train_start, 2),
        "summary": summary,
        "files": {
            "train_results_csv": train_csv.as_posix() if train_csv else None,
            "events_jsonl": events_path.as_posix() if rows else None,
        },
        "weights_sha256": checksums,
    }
    with open(artifacts_dir / "training_process.json", "w", encoding="utf-8") as f:
        json.dump(proc_info, f, indent=2)

    # Snapshot data.yaml
    copy_if_exists(args.data, artifacts_dir)

    # -------- ONNX Export (from BEST weights if available) --------
    export_info = {}
    if args.export_onnx:
        print("[bold cyan]Exporting ONNX...[/]")
        export_dir = artifacts_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        export_source = best_pt if best_pt and best_pt.exists() else (last_pt if last_pt and last_pt.exists() else None)
        if export_source is None:
            # fallback: export current in-memory model
            export_model = model
            export_tag = "in_memory_model"
        else:
            export_model = YOLO(export_source.as_posix())
            export_tag = "best" if export_source == best_pt else "last"

        onnx_path = None
        try:
            # Ultralytics handles additional kwargs like opset/dynamic/simplify where supported
            export_results = export_model.export(
                format="onnx",
                opset=args.export_opset,
                dynamic=args.export_dynamic,
                simplify=args.export_simplify,
                project=export_dir.as_posix(),
                name=f"{run_name}_{export_tag}",
            )
            # export() returns a path or namespace depending on version; normalize
            # Find newest *.onnx in export_dir
            cand = sorted(export_dir.glob("**/*.onnx"))
            if cand:
                onnx_path = cand[-1]
                export_info = {
                    "onnx_path": onnx_path.as_posix(),
                    "onnx_sha256": sha256sum(onnx_path),
                    "opset": args.export_opset,
                    "dynamic": bool(args.export_dynamic),
                    "simplify": bool(args.export_simplify),
                    "export_source": export_tag,
                }
        except Exception as e:
            export_info = {"error": f"ONNX export failed: {e!r}"}

        with open(artifacts_dir / "export_manifest.json", "w", encoding="utf-8") as f:
            json.dump(export_info, f, indent=2)

    # Final manifest for convenience
    with open(artifacts_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            **manifest,
            "training_process": (artifacts_dir / "training_process.json").as_posix(),
            "export": export_info or None,
        }, f, indent=2)

    # Done
    print("\n[bold green]Done! Key outputs:[/]")
    print(f"  • Best model: {manifest['train_files'].get('best.pt')}")
    print(f"  • Training CSV/plots: {manifest['train_files'].get('results.csv')}")
    print(f"  • Validation CSV/plots: {manifest['val_files'].get('results.csv')}")
    if test_run_dir:
        print(f"  • Test CSV/plots: {manifest['test_files'].get('results.csv')}")
    if export_info.get("onnx_path"):
        print(f"  • ONNX: {export_info['onnx_path']} (sha256={export_info['onnx_sha256'][:12]}...)")
    print(f"  • Artifacts folder: {artifacts_dir.as_posix()}")
    print("    - training_process.json (system/env, args, tuned hypers, summary)")
    print("    - train_events.jsonl (per-epoch metrics)")
    print("    - export_manifest.json (ONNX details)")
    print("Tip: load events JSONL to chart loss/metrics over epochs in your reporting pipeline.")

if __name__ == "__main__":
    main()
