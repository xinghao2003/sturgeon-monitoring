import threading
import time
import cv2
import os
import numpy as np
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, filedialog, Scale, HORIZONTAL, StringVar,
    OptionMenu, Checkbutton, IntVar, Listbox, EXTENDED, END, BooleanVar, Canvas
)
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

from ultralytics import YOLO

# -----------------------------
# Basic settings
# -----------------------------
DEFAULT_VIDEO = "sample.mp4"   # Put a mock video in the same folder or pick via dialog
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_TRACKER = "botsort.yaml"  # or "bytetrack.yaml"
GUI_REFRESH_MS = 30               # GUI refresh cadence (main thread .after loop)

# Available OpenCV colormaps
CV_COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "HOT": cv2.COLORMAP_HOT,
    "BONE": cv2.COLORMAP_BONE,
    "VIRIDIS": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
}

class YOLOTrackerApp:
    def __init__(self, master: Tk):
        self.master = master
        self.master.title("YOLOv11 Tracker (PT / ONNX)")

        # Try to maximize and center
        self._maximize_and_center()

        # App state
        self.model = None
        self.model_path = None
        self.video_path = Path(DEFAULT_VIDEO) if Path(DEFAULT_VIDEO).exists() else None
        self.running = False
        self.worker_thread = None
        self.stop_event = threading.Event()

        # Shared data between worker and GUI
        self.latest_frame = None      # PIL.Image (RGB)
        self.readout_text = StringVar(value="FPS: 0.0 | Frame: - | Model: - | Tracker: - | Device: auto")
        self.status_text = StringVar(value="Status: idle")
        self.eta_text = StringVar(value="ETA: --:-- | 0/0")
        self.device_var = StringVar(value="auto")
        self.tracker_var = StringVar(value=DEFAULT_TRACKER)
        self.show_ids_var = IntVar(value=1)
        self.save_video_var = IntVar(value=0)

        # Heatmap settings/state
        self.heat_overlay_var = IntVar(value=1)         # overlay on annotated video feed
        self.heat_export_var = IntVar(value=0)          # export heatmap-only video
        self.per_class_export_var = IntVar(value=0)     # export per-class heatmaps
        self.heat_opacity = 0.5
        self.heat_opacity_var = IntVar(value=int(self.heat_opacity * 100))
        self.sigma_factor_var = IntVar(value=25)        # 10..100 -> 0.10..1.00
        self.percentile_var = IntVar(value=99)          # 90..100
        self.colormap_var = StringVar(value="JET")
        self.decay_enabled = BooleanVar(value=False)
        self.decay_rate_var = IntVar(value=0)           # 0..20 (% per frame)
        self.max_det_var = IntVar(value=120)

        self.heat_accum = None                          # float32 (H,W)
        self.heat_writer = None
        self.heat_writer_path = None

        # Per-class accumulators/writers
        self.heat_accum_per_cls = {}    # cls_id -> np.ndarray (H,W)
        self.heat_writer_per_cls = {}   # cls_id -> cv2.VideoWriter
        self.heat_writer_path_per_cls = {}  # cls_id -> path

        # Classes UI state
        self.class_listbox = None
        self.model_class_names = []   # list[str]
        self.model_class_ids = []     # list[int]

        # In/out video writer
        self.writer = None
        self.writer_path = None
        self.writer_fps = 30
        self.writer_size = None

        # Progress
        self.total_frames = 0
        self.current_frame_index = 0

        # ROI state (in FRAME coordinates)
        self.roi_enabled = BooleanVar(value=False)
        self.roi_strict = BooleanVar(value=False)   # mask outside ROI before detection
        self.roi_edit_mode = False
        self.roi_points = []         # list[(x,y)] in FRAME coords
        self._last_fit = None        # mapping from widget coords -> frame coords
        self.video_label_bindings_set = False

        # Build UI
        self._build_layout()

        # GUI update loop
        self._update_gui()

    # ---------------- UI layout ----------------
    def _build_layout(self):
        root_pane = ttk.Panedwindow(self.master, orient="horizontal")
        root_pane.pack(fill="both", expand=True)

        # Left (video)
        self.left_frame = Frame(root_pane, bg="#000000")
        root_pane.add(self.left_frame, weight=3)
        self.video_label = Label(self.left_frame, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        # Right (controls) - scrollable container with max width of 1/3
        right_container = Frame(root_pane)
        root_pane.add(right_container, weight=1)
        
        # Create canvas and scrollbar for right panel
        canvas = Canvas(right_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        self.right_frame = Frame(canvas)
        self.right_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.right_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Readout (top)
        readout_frame = Frame(self.right_frame)
        readout_frame.pack(fill="x", padx=8, pady=(8, 4))
        Label(readout_frame, textvariable=self.readout_text, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # Progress
        progress_frame = Frame(self.right_frame)
        progress_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x")
        Label(progress_frame, textvariable=self.eta_text).pack(anchor="w")

        # Device + Tracker row
        dt_row = Frame(self.right_frame)
        dt_row.pack(fill="x", padx=8, pady=6)
        Label(dt_row, text="Device:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        OptionMenu(dt_row, self.device_var, "auto", "cpu", "cuda", "intel:gpu").grid(row=0, column=1, padx=4, pady=2, sticky="w")
        Label(dt_row, text="Tracker:").grid(row=0, column=2, padx=12, pady=2, sticky="w")
        OptionMenu(dt_row, self.tracker_var, "botsort.yaml", "bytetrack.yaml").grid(row=0, column=3, padx=4, pady=2, sticky="w")

        # Buttons row
        btns_row = Frame(self.right_frame)
        btns_row.pack(fill="x", padx=8, pady=6)
        Button(btns_row, text="Load Model (.pt/.onnx)", command=self.load_model).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        Button(btns_row, text="Pick MP4", command=self.pick_video).grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.start_btn = Button(btns_row, text="Start", command=self.start_tracking, state="disabled")
        self.start_btn.grid(row=0, column=2, padx=4, pady=2, sticky="w")
        self.stop_btn = Button(btns_row, text="Stop", command=self.stop_tracking, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=4, pady=2, sticky="w")

        # Inference sliders
        try:
            from tkinter import LabelFrame
        except Exception:
            class LabelFrame(Frame):
                def __init__(self, master=None, text="", **kw):
                    super().__init__(master, **kw)
                    Label(self, text=text).pack(anchor="w")

        sliders_frame = LabelFrame(self.right_frame, text="Inference", padx=8, pady=8)
        sliders_frame.pack(fill="x", padx=8, pady=4)
        self.conf_scale = Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.01,
                                orient=HORIZONTAL, label="Confidence", length=280)
        self.conf_scale.set(DEFAULT_CONF)
        self.conf_scale.pack(anchor="w")
        self.iou_scale = Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.01,
                               orient=HORIZONTAL, label="IoU", length=280)
        self.iou_scale.set(DEFAULT_IOU)
        self.iou_scale.pack(anchor="w")
        self.max_det_scale = Scale(sliders_frame, from_=20, to=300, resolution=10,
                            orient=HORIZONTAL, label="Max detections", length=280,
                            variable=self.max_det_var)
        self.max_det_scale.pack(anchor="w")

        # Class filtering
        cls_frame = LabelFrame(self.right_frame, text="Class Filter", padx=8, pady=8)
        cls_frame.pack(fill="both", expand=False, padx=8, pady=4)
        Label(cls_frame, text="(Select one or more to filter; empty = all classes)").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.class_listbox = Listbox(cls_frame, selectmode=EXTENDED, height=8, exportselection=False)
        self.class_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        cls_frame.grid_rowconfigure(1, weight=1)
        cls_frame.grid_columnconfigure(0, weight=1)
        Button(cls_frame, text="Select All", command=self._cls_select_all).grid(row=2, column=0, sticky="w", pady=6)
        Button(cls_frame, text="Clear", command=self._cls_clear).grid(row=2, column=1, sticky="e", pady=6)

        # Toggles
        toggles_frame = LabelFrame(self.right_frame, text="Options", padx=8, pady=8)
        toggles_frame.pack(fill="x", padx=8, pady=4)
        Checkbutton(toggles_frame, text="Show track IDs", variable=self.show_ids_var).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        Checkbutton(toggles_frame, text="Save output video", variable=self.save_video_var).grid(row=0, column=1, padx=4, pady=2, sticky="w")

        # Heatmap controls
        heat_frame = LabelFrame(self.right_frame, text="Heatmap", padx=8, pady=8)
        heat_frame.pack(fill="x", padx=8, pady=4)
        Checkbutton(heat_frame, text="Enable overlay", variable=self.heat_overlay_var).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        Checkbutton(heat_frame, text="Export heatmap-only video", variable=self.heat_export_var).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        Checkbutton(heat_frame, text="Per-class heatmap export", variable=self.per_class_export_var).grid(row=0, column=2, sticky="w", padx=4, pady=2)

        Label(heat_frame, text="Overlay opacity").grid(row=1, column=0, sticky="w", padx=4)
        Scale(heat_frame, from_=0, to=100, orient=HORIZONTAL, length=220,
              command=self._on_opacity_change, variable=self.heat_opacity_var).grid(row=1, column=1, sticky="w", padx=4)

        Label(heat_frame, text="Sigma factor (x box size)").grid(row=2, column=0, sticky="w", padx=4)
        Scale(heat_frame, from_=10, to=100, orient=HORIZONTAL, length=220,
              variable=self.sigma_factor_var).grid(row=2, column=1, sticky="w", padx=4)

        Label(heat_frame, text="Percentile cap (contrast)").grid(row=3, column=0, sticky="w", padx=4)
        Scale(heat_frame, from_=90, to=100, orient=HORIZONTAL, length=220,
              variable=self.percentile_var).grid(row=3, column=1, sticky="w", padx=4)

        Label(heat_frame, text="Colormap").grid(row=4, column=0, sticky="w", padx=4)
        OptionMenu(heat_frame, self.colormap_var, *CV_COLORMAPS.keys()).grid(row=4, column=1, sticky="w", padx=4)

        Checkbutton(heat_frame, text="Temporal decay", variable=self.decay_enabled).grid(row=5, column=0, sticky="w", padx=4)
        Label(heat_frame, text="Decay per frame (%)").grid(row=5, column=1, sticky="w", padx=4)
        Scale(heat_frame, from_=0, to=20, orient=HORIZONTAL, length=180,
              variable=self.decay_rate_var).grid(row=5, column=2, sticky="w", padx=4)

        Button(heat_frame, text="Reset heatmap", command=self._reset_heatmap).grid(row=6, column=0, sticky="w", padx=4, pady=(6, 0))

        # ROI controls
        roi_frame = LabelFrame(self.right_frame, text="ROI (Polygon)", padx=8, pady=8)
        roi_frame.pack(fill="x", padx=8, pady=4)
        Checkbutton(roi_frame, text="ROI Enabled", variable=self.roi_enabled).grid(row=0, column=0, sticky="w", padx=4)
        Checkbutton(roi_frame, text="Strict ROI (mask outside before detection)", variable=self.roi_strict).grid(row=0, column=1, sticky="w", padx=4)
        Button(roi_frame, text="Start/Edit ROI", command=self._roi_start_edit).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        Button(roi_frame, text="Finish ROI", command=self._roi_finish).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        Button(roi_frame, text="Clear ROI", command=self._roi_clear).grid(row=1, column=2, sticky="w", padx=4, pady=2)
        Label(roi_frame, text="Tip: Click on video to add points. Finish to close polygon.").grid(row=2, column=0, columnspan=3, sticky="w", padx=4)

        # Status
        status_frame = Frame(self.right_frame)
        status_frame.pack(fill="x", padx=8, pady=(6, 10))
        Label(status_frame, textvariable=self.status_text).pack(anchor="w")

        # First placeholder & bindings
        self._render_placeholder("No video loaded.\nClick 'Pick MP4' to choose a file.")
        self._ensure_video_label_bindings()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------- Window helpers --------------
    def _maximize_and_center(self):
        try:
            self.master.state("zoomed")  # Windows
        except Exception:
            try:
                self.master.attributes("-zoomed", True)  # Some *nix
            except Exception:
                self.master.update_idletasks()
                w, h = 1280, 800
                sw = self.master.winfo_screenwidth()
                sh = self.master.winfo_screenheight()
                x = int((sw - w) / 2)
                y = int((sh - h) / 2)
                self.master.geometry(f"{w}x{h}+{x}+{y}")

    # ---------------- Actions ----------------
    def load_model(self):
        device = self.device_var.get()
        
        # For Intel GPU, allow folder selection; otherwise, file selection
        if device == "intel:gpu":
            path = filedialog.askdirectory(title="Select YOLO model folder (for Intel GPU)")
            if not path:
                return
            self.model_path = path
        else:
            path = filedialog.askopenfilename(
                title="Select YOLO model (.pt or .onnx)",
                filetypes=[("YOLO model", "*.pt *.pt *.onnx"), ("All files", "*.*")]
            )
            if not path:
                return
            self.model_path = path
        
        self.status_text.set("Loading model…")
        self.master.update_idletasks()
        try:
            # Normalize device string for YOLO/OpenVINO compatibility
            yolo_device = self._normalize_device(device)
            
            self.model = YOLO(self.model_path, task="segment")
            
            # Move to device ONLY if not auto (avoids double initialization)
            if device != "auto":
                try:
                    self.model.to(yolo_device)
                except Exception as dev_err:
                    self.status_text.set(f"Device move to '{yolo_device}' failed: {dev_err}. Using auto.")
                    yolo_device = "auto"
            
            self.status_text.set(f"Loaded model: {os.path.basename(self.model_path)}")
            self._populate_class_list()
            self._update_start_state()
        except Exception as e:
            self.model = None
            self.model_path = None
            self.model_class_names = []
            self.model_class_ids = []
            self._refresh_class_listbox()
            self.status_text.set(f"Failed to load model: {e}")
            self._update_start_state()

    def _normalize_device(self, device_str: str) -> str:
        """
        Convert UI device string to YOLO/OpenVINO compatible format.
        """
        if device_str == "auto":
            return "auto"
        elif device_str == "cpu":
            return "cpu"
        elif device_str == "cuda":
            return "cuda"
        elif device_str == "intel:gpu":
            return "GPU.0"  # OpenVINO GPU format
        else:
            return device_str

    def pick_video(self):
        path = filedialog.askopenfilename(
            title="Select MP4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")]
        )
        if not path:
            return
        self.video_path = Path(path)
        self.status_text.set(f"Video: {self.video_path.name}")
        self._update_start_state()

    def _update_start_state(self):
        ok = (self.model is not None) and (self.video_path and self.video_path.exists())
        self.start_btn.config(state=("normal" if ok else "disabled"))

    # ----- classes UI -----
    def _populate_class_list(self):
        names = []
        try:
            raw = getattr(self.model, "names", None)
            if raw is None:
                raw = getattr(getattr(self.model, "model", None), "names", None)
            if isinstance(raw, dict):
                ids = sorted(raw.keys())
                names = [raw[i] for i in ids]
                self.model_class_ids = ids
                self.model_class_names = names
            elif isinstance(raw, list):
                self.model_class_ids = list(range(len(raw)))
                self.model_class_names = list(raw)
            else:
                self.model_class_ids = []
                self.model_class_names = []
        except Exception:
            self.model_class_ids = []
            self.model_class_names = []
        self._refresh_class_listbox()

    def _refresh_class_listbox(self):
        self.class_listbox.delete(0, END)
        if not self.model_class_names:
            self.class_listbox.insert(END, "(Model has no class list)")
            self.class_listbox.config(state="disabled")
        else:
            self.class_listbox.config(state="normal")
            for i, name in zip(self.model_class_ids, self.model_class_names):
                self.class_listbox.insert(END, f"{i}: {name}")

    def _cls_select_all(self):
        if not self.model_class_names:
            return
        self.class_listbox.select_set(0, END)

    def _cls_clear(self):
        self.class_listbox.selection_clear(0, END)

    def _selected_class_ids(self):
        if not self.model_class_names:
            return None
        sels = self.class_listbox.curselection()
        if len(sels) == 0 or len(sels) == len(self.model_class_names):
            return None
        return [self.model_class_ids[idx] for idx in sels]

    # ----- ROI UI helpers -----
    def _roi_start_edit(self):
        if self.running:
            self.status_text.set("Stop the video to edit ROI.")
            return
        self.roi_edit_mode = True
        self.status_text.set("ROI edit: click on the video to add points…")

    def _roi_finish(self):
        if not self.roi_points or len(self.roi_points) < 3:
            self.status_text.set("ROI needs at least 3 points.")
            return
        self.roi_edit_mode = False
        self.status_text.set("ROI closed.")

    def _roi_clear(self):
        self.roi_points = []
        self.roi_edit_mode = False
        self.status_text.set("ROI cleared.")

    def _ensure_video_label_bindings(self):
        if self.video_label_bindings_set:
            return
        self.video_label.bind("<Button-1>", self._on_video_click)
        self.video_label_bindings_set = True

    def _on_video_click(self, event):
        if not self.roi_edit_mode:
            return
        if self.latest_frame is None:
            return
        # Map widget coords -> frame coords
        frame_pt = self._widget_to_frame_coords(event.x, event.y)
        if frame_pt is None:
            return
        self.roi_points.append(frame_pt)
        self.status_text.set(f"ROI point added: {frame_pt}. Points: {len(self.roi_points)}")

    def _widget_to_frame_coords(self, wx, wy):
        """
        Convert widget click (wx,wy) to frame coordinates based on last fit.
        """
        lf = self._last_fit
        if lf is None:
            return None
        offx, offy = lf["offx"], lf["offy"]
        imgw, imgh = lf["imgw"], lf["imgh"]
        framew, frameh = lf["framew"], lf["frameh"]
        if wx < offx or wy < offy or wx >= offx + imgw or wy >= offy + imgh:
            return None
        # scale from displayed image to frame size
        sx = framew / imgw
        sy = frameh / imgh
        fx = int((wx - offx) * sx)
        fy = int((wy - offy) * sy)
        return (fx, fy)

    # ----- heatmap helpers -----
    def _on_opacity_change(self, _val):
        self.heat_opacity = float(self.heat_opacity_var.get()) / 100.0

    def _reset_heatmap(self):
        self.heat_accum = None
        self.heat_accum_per_cls = {}
        self.status_text.set("Heatmap(s) reset.")

    # ----- run/stop -----
    def start_tracking(self):
        if self.running:
            return
        if not self.model:
            self.status_text.set("Load a model first.")
            return
        if not self.video_path or not self.video_path.exists():
            self.status_text.set("Pick an MP4 first.")
            return

        # Device handling
        device = self.device_var.get()
        applied_device = self._normalize_device(device)
        
        try:
            if device != "auto":
                self.model.to(applied_device)
        except Exception as e:
            self.status_text.set(f"Device set failed ({applied_device}), using auto. ({e})")
            applied_device = "auto"
        
        self._set_readout_device(applied_device)

        self.stop_event.clear()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_text.set("Tracking… (press Stop to end)")

        # Reset writers & progress & heatmap
        self._close_writer()
        self._close_heat_writer()
        self._close_per_class_writers()
        self.total_frames = 0
        self.current_frame_index = 0
        self.progress["value"] = 0
        self.progress["maximum"] = 1
        self.eta_text.set("ETA: --:-- | 0/0")
        self.heat_accum = None
        self.heat_accum_per_cls = {}

        # Launch worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop_tracking(self):
        self.stop_event.set()
        self.running = False
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.status_text.set("Stopping…")

    # ---------------- Worker thread ----------------
    def _worker_loop(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.status_text.set("Failed to open video.")
            self.running = False
            return

        # Progress info
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
        self.total_frames = total
        self.progress["maximum"] = max(1, total)

        tracker_cfg = self.tracker_var.get()
        conf = float(self.conf_scale.get())
        iou = float(self.iou_scale.get())
        max_det = max(1, int(self.max_det_var.get()))
        model_name = os.path.basename(self.model_path) if self.model_path else "-"
        selected_classes = self._selected_class_ids()

        # fps compute
        fps_calc_window_start = time.time()
        frames_in_window = 0
        live_fps = 0.0

        start_time = time.time()
        need_writer = bool(self.save_video_var.get())
        need_heat_writer = bool(self.heat_export_var.get())
        need_per_class = bool(self.per_class_export_var.get())

        try:
            while not self.stop_event.is_set() and cap.isOpened():
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # Read index for progress
                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.current_frame_index = idx

                # Strict ROI masking (before detection)
                roi_poly = np.array(self.roi_points, dtype=np.int32) if len(self.roi_points) >= 3 else None
                if self.roi_enabled.get() and self.roi_strict.get() and roi_poly is not None:
                    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [roi_poly], 255)
                    frame_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

                # Run tracking with normalized device
                applied_device = self._normalize_device(self.device_var.get())
                results = self.model.track(
                    source=frame_bgr,
                    persist=True,
                    tracker=tracker_cfg,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    classes=selected_classes,
                    verbose=True,  # Changed to False to reduce spam
                    device=applied_device,
                )
                res = results[0]

                # ---- Heatmap accumulation ----
                h, w = frame_bgr.shape[:2]
                if self.heat_accum is None:
                    self.heat_accum = np.zeros((h, w), dtype=np.float32)
                if need_per_class:
                    for cid in self.model_class_ids:
                        if cid not in self.heat_accum_per_cls:
                            self.heat_accum_per_cls[cid] = np.zeros((h, w), dtype=np.float32)

                # Temporal decay
                if self.decay_enabled.get():
                    rate = float(self.decay_rate_var.get()) / 100.0
                    decay = max(0.0, min(rate, 0.95))
                    self.heat_accum *= (1.0 - decay)
                    if need_per_class:
                        for cid in list(self.heat_accum_per_cls.keys()):
                            self.heat_accum_per_cls[cid] *= (1.0 - decay)

                # Extract boxes (xyxy) and (optionally) class ids
                boxes = getattr(res, "boxes", None)
                cls_arr = None
                if boxes is not None:
                    if getattr(boxes, "xyxy", None) is not None:
                        xyxy = boxes.xyxy.cpu().numpy()
                    else:
                        xyxy = None
                    if getattr(boxes, "cls", None) is not None:
                        cls_arr = boxes.cls.int().cpu().numpy()
                    else:
                        cls_arr = None
                else:
                    xyxy = None

                # ROI filtering after detection (center-in-poly)
                if xyxy is not None:
                    # If ROI enabled with polygon, keep only boxes whose centers are inside
                    if self.roi_enabled.get() and len(self.roi_points) >= 3:
                        roi_poly = np.array(self.roi_points, dtype=np.int32)
                        keep = []
                        for i, (x1, y1, x2, y2) in enumerate(xyxy):
                            cx = int(0.5 * (x1 + x2))
                            cy = int(0.5 * (y1 + y2))
                            if cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0:
                                keep.append(i)
                        if keep:
                            xyxy = xyxy[keep]
                            if cls_arr is not None:
                                cls_arr = cls_arr[keep]
                        else:
                            xyxy = None
                            cls_arr = None

                # Add Gaussians
                if xyxy is not None:
                    sigma_factor = max(0.10, min(1.0, float(self.sigma_factor_var.get()) / 100.0))
                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        cx = int(0.5 * (x1 + x2))
                        cy = int(0.5 * (y1 + y2))
                        bw = max(int(x2 - x1), 1)
                        bh = max(int(y2 - y1), 1)
                        sigma = max(3, int(min(bw, bh) * sigma_factor))
                        self._splat_gaussian(self.heat_accum, cx, cy, sigma)
                        if need_per_class and cls_arr is not None:
                            cid = int(cls_arr[i])
                            if cid in self.heat_accum_per_cls:
                                self._splat_gaussian(self.heat_accum_per_cls[cid], cx, cy, sigma)

                # Annotated frame
                annotated_bgr = res.plot()

                # Draw IDs
                if self.show_ids_var.get() and boxes is not None and getattr(boxes, "id", None) is not None and getattr(boxes, "xyxy", None) is not None:
                    ids = boxes.id.int().cpu().tolist()
                    bxy = boxes.xyxy.int().cpu().tolist()
                    for (x1, y1, x2, y2), tid in zip(bxy, ids):
                        cv2.putText(annotated_bgr, f"ID {tid}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw ROI polygon overlay
                if len(self.roi_points) >= 2:
                    pts = np.array(self.roi_points, dtype=np.int32)
                    cv2.polylines(annotated_bgr, [pts], isClosed=(not self.roi_edit_mode and len(self.roi_points) >= 3),
                                  color=(0, 255, 255), thickness=2)

                # Build heatmap RGB (normalized colormap)
                heat_rgb = self._heatmap_rgb(self.heat_accum)

                # Overlay onto annotated video if enabled
                if self.heat_overlay_var.get():
                    alpha = self.heat_opacity
                    annotated_bgr = cv2.addWeighted(annotated_bgr, 1.0, cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR), alpha, 0)

                # Convert to RGB PIL for GUI
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                self.latest_frame = Image.fromarray(annotated_rgb)

                # Init writers on first frame if needed
                if need_writer and self.writer is None:
                    self._init_main_writer(cap, annotated_bgr)
                if need_heat_writer and self.heat_writer is None:
                    self._init_heat_writer(cap, heat_rgb)
                if need_per_class:
                    self._init_per_class_writers_if_needed(cap, heat_rgb.shape[1], heat_rgb.shape[0])

                # Write frames
                if self.writer is not None:
                    self.writer.write(annotated_bgr)
                if self.heat_writer is not None:
                    self.heat_writer.write(cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
                if need_per_class and self.heat_writer_per_cls:
                    for cid, acc in self.heat_accum_per_cls.items():
                        rgb = self._heatmap_rgb(acc)
                        self.heat_writer_per_cls[cid].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                # FPS calc
                frames_in_window += 1
                now = time.time()
                if now - fps_calc_window_start >= 0.5:
                    live_fps = frames_in_window / (now - fps_calc_window_start)
                    fps_calc_window_start = now
                    frames_in_window = 0

                # Readout + progress + ETA
                frame_h, frame_w = annotated_bgr.shape[:2]
                dev_str = self.device_var.get()
                self.readout_text.set(
                    f"FPS: {live_fps:.1f} | Frame: {frame_w}x{frame_h} | Model: {model_name} | Tracker: {tracker_cfg} | Device: {dev_str}"
                )

                if self.total_frames > 0 and live_fps > 0:
                    frames_done = min(self.current_frame_index, self.total_frames)
                    frames_left = max(self.total_frames - frames_done, 0)
                    eta_sec = frames_left / max(live_fps, 1e-6)
                    eta_str = self._fmt_time(eta_sec)
                    self.progress["value"] = frames_done
                    self.eta_text.set(f"ETA: {eta_str} | {frames_done}/{self.total_frames}")
                else:
                    self.progress["value"] = 0 if self.total_frames <= 0 else min(self.current_frame_index, self.total_frames)
                    elapsed = time.time() - start_time
                    self.eta_text.set(f"Elapsed: {self._fmt_time(elapsed)} | {self.current_frame_index}/{self.total_frames or '?'}")

        except Exception as e:
            self.status_text.set(f"Runtime error: {e}")
        finally:
            cap.release()
            self._close_writer()
            self._close_heat_writer()
            self._close_per_class_writers()
            self.running = False
            self.stop_btn.config(state="disabled")
            self.start_btn.config(state="normal")

            saved = []
            if self.writer_path and need_writer:
                saved.append(os.path.basename(self.writer_path))
            if self.heat_writer_path and need_heat_writer:
                saved.append(os.path.basename(self.heat_writer_path))
            if self.heat_writer_path_per_cls and self.per_class_export_var.get():
                saved += [os.path.basename(p) for p in self.heat_writer_path_per_cls.values()]
            if saved:
                self.status_text.set("Stopped. Saved: " + ", ".join(saved))
            else:
                self.status_text.set("Stopped.")

    # ---------------- GUI main-thread loop ----------------
    def _update_gui(self):
        if self.latest_frame is not None:
            disp, fit = self._fit_to_widget(self.latest_frame, self.video_label)
            self._last_fit = fit
            self._tk_img = ImageTk.PhotoImage(image=disp)
            self.video_label.config(image=self._tk_img)
        else:
            self._render_placeholder("No video feed.\nPress Start to begin.")
        self.master.after(GUI_REFRESH_MS, self._update_gui)

    # ---------------- Helpers ----------------
    def _fit_to_widget(self, pil_img: Image.Image, widget: Label):
        widget.update_idletasks()
        w = max(widget.winfo_width(), 320)
        h = max(widget.winfo_height(), 240)
        img = pil_img.copy()
        img.thumbnail((w, h))  # keeps aspect ratio
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        x = (w - img.width) // 2
        y = (h - img.height) // 2
        canvas.paste(img, (x, y))
        fit = {
            "offx": x, "offy": y,
            "imgw": img.width, "imgh": img.height,
            "framew": pil_img.width, "frameh": pil_img.height
        }
        return canvas, fit

    def _render_placeholder(self, message: str):
        self.video_label.update_idletasks()
        w = max(self.video_label.winfo_width(), 640)
        h = max(self.video_label.winfo_height(), 360)
        img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        lines = message.split("\n")
        line_h = 24
        total_h = line_h * len(lines)
        y_cursor = (h - total_h) // 2
        for line in lines:
            text_w = draw.textlength(line)
            x = int((w - text_w) / 2)
            draw.text((x, y_cursor), line, fill=(200, 200, 200))
            y_cursor += line_h
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tk_img)

    def _close_writer(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None
        self.writer_path = None

    def _init_main_writer(self, cap, annotated_bgr):
        h, w = annotated_bgr.shape[:2]
        self.writer_size = (w, h)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        self.writer_fps = src_fps if src_fps and src_fps > 0 else 30.0
        out_name = f"{self.video_path.stem}_tracked.mp4"
        self.writer_path = str(self.video_path.with_name(out_name))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.writer_path, fourcc, self.writer_fps, self.writer_size)

    def _close_heat_writer(self):
        if self.heat_writer is not None:
            try:
                self.heat_writer.release()
            except Exception:
                pass
        self.heat_writer = None
        self.heat_writer_path = None

    def _init_heat_writer(self, cap, heat_rgb):
        h, w = heat_rgb.shape[:2]
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = src_fps if src_fps and src_fps > 0 else 30.0
        out_name = f"{self.video_path.stem}_heatmap.mp4"
        self.heat_writer_path = str(self.video_path.with_name(out_name))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.heat_writer = cv2.VideoWriter(self.heat_writer_path, fourcc, fps, (w, h))

    def _close_per_class_writers(self):
        for wri in self.heat_writer_per_cls.values():
            try:
                wri.release()
            except Exception:
                pass
        self.heat_writer_per_cls.clear()
        self.heat_writer_path_per_cls.clear()

    def _init_per_class_writers_if_needed(self, cap, w, h):
        if self.heat_writer_per_cls:
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = src_fps if src_fps and src_fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Only create writers for classes currently selected (or all if none selected)
        active_ids = self._selected_class_ids()
        if active_ids is None:  # means "all"
            active_ids = self.model_class_ids
        for cid in active_ids:
            label = str(cid)
            if 0 <= cid < len(self.model_class_names):
                label = f"{cid}-{self.model_class_names[cid]}"
            out_name = f"{self.video_path.stem}_heatmap_cls-{label}.mp4"
            p = str(self.video_path.with_name(out_name))
            self.heat_writer_path_per_cls[cid] = p
            self.heat_writer_per_cls[cid] = cv2.VideoWriter(p, fourcc, fps, (w, h))

    def _on_close(self):
        self.stop_event.set()
        self.running = False
        time.sleep(0.05)
        self._close_writer()
        self._close_heat_writer()
        self._close_per_class_writers()
        self.master.destroy()

    def _fmt_time(self, seconds: float) -> str:
        seconds = int(max(0, seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    def _set_readout_device(self, dev: str):
        parts = self.readout_text.get().split("|")
        if len(parts) >= 5:
            parts[-1] = f" Device: {dev}"
            self.readout_text.set(" |".join(parts))
        else:
            self.readout_text.set(self.readout_text.get() + f" | Device: {dev}")

    # ---------- Heatmap math ----------
    def _splat_gaussian(self, acc: np.ndarray, cx: int, cy: int, sigma: int):
        if sigma <= 0:
            return
        h, w = acc.shape[:2]
        r = int(3 * sigma)
        x0 = max(cx - r, 0); x1 = min(cx + r + 1, w)
        y0 = max(cy - r, 0); y1 = min(cy + r + 1, h)
        if x1 <= x0 or y1 <= y0:
            return
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma)).astype(np.float32)
        s = g.sum()
        if s > 1e-6:
            g /= s
        acc[y0:y1, x0:x1] += g

    def _heatmap_rgb(self, acc: np.ndarray) -> np.ndarray:
        if acc is None or acc.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        pctl = max(90, min(100, int(self.percentile_var.get())))
        p = np.percentile(acc, pctl)
        mx = max(p, 1e-6)
        norm = np.clip(acc / mx, 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        cmap_id = CV_COLORMAPS.get(self.colormap_var.get(), cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(gray, cmap_id)       # BGR
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)   # RGB
        return rgb

# --------------- main ---------------
if __name__ == "__main__":
    root = Tk()
    app = YOLOTrackerApp(root)
    root.mainloop()
