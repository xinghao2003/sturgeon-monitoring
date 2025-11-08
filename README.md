# Sturgeon Monitoring - YOLO-based Real-time Instance Segmentation

A comprehensive sturgeon monitoring system using YOLOv11 instance segmentation with real-time tracking, behavior analysis, and heatmap visualization. Features a desktop GUI application for live monitoring and training infrastructure for custom model development.

## Features

- **Real-time Instance Segmentation**: YOLOv11-based sturgeon detection and instance segmentation
- **Multi-Format Model Support**: PyTorch, ONNX, OpenVINO, TensorRT, and TorchScript models
- **Live Object Tracking**: Continuous tracking of individual sturgeons with ID persistence
- **Behavior Analysis**: Automated detection of behavioral anomalies:
  - Lethargy/Activity Drop detection (mean displacement analysis)
  - Crowding detection (spatial clustering analysis)
  - Edge/Wall-Pacing detection (perimeter band analysis)
  - Inflow Magnet detection (ROI occupancy analysis)
- **Heatmap Visualization**: Real-time occupancy heatmaps with configurable decay and intensity
- **Alert System**: Configurable alerts based on behavior thresholds with z-score analysis
- **Performance Monitoring**: Real-time FPS tracking and resource utilization
- **Model Training**: Full hyperparameter tuning, training, validation, and export pipeline
- **Logging & Artifacts**: Comprehensive training artifacts, metrics, and system information capture

## System Requirements

- **Python**: 3.12+
- **GPU** (optional but recommended): NVIDIA CUDA 13.0+ with cuDNN
- **OS**: Windows, Linux, macOS
- **RAM**: 8GB+ recommended
- **VRAM**: 2GB+ for GPU acceleration

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sturgeon-monitoring
```

### 2. Install Dependencies

#### Option A: Using `uv` (recommended, faster)

```bash
uv sync
```

#### Option B: Using pip with PyTorch CUDA 13.0

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

#### Option C: Using pip without GPU

```bash
pip install -r requirements.txt
```

### 3. Prepare Model Files

Place your trained YOLO model in the `models/` directory:

```
models/
├── best.pt              # PyTorch format (for app.py)
├── best.onnx            # ONNX format (cross-platform)
├── best.engine          # TensorRT format (NVIDIA GPUs)
├── best.torchscript     # TorchScript format
└── best_openvino_model/ # OpenVINO optimized model
    ├── best.xml
    └── metadata.yaml
```

## Usage

### Running the Application

Start the real-time monitoring GUI:

```bash
python app.py
```

The application will open with:

- **Left Panel**: Video feed and heatmap visualization
- **Bottom Left**: Activity logs and behavior alerts
- **Bottom Center**: Real-time behavior metrics
- **Right Panel**: Configuration controls

### Application Configuration

#### Input Source Options

Configure the input source in the GUI or modify `app.py`:

**Camera/Virtual Camera (OBS)**

```
Input Source: 0  (or 1, 2, etc. for multiple cameras)
```

For virtual camera using OBS Studio:

1. Open OBS Studio
2. Create a virtual camera (Tools → VirtualCamera)
3. Set Input Source to the camera index (typically `0` for default)
4. Start streaming in OBS before launching app.py

**Video File**

```
Input Source: /path/to/video.mp4
```

**RTSP Stream**

```
Input Source: rtsp://user:password@host:554/stream
```

**HTTP Stream**

```
Input Source: https://example.com/stream.m3u8
```

#### Detection Parameters

- **Model Path**: Select from trained models in `models/` folder
- **Confidence**: Detection confidence threshold (0.0-1.0, default: 0.50)
- **IoU Threshold**: Non-Maximum Suppression IoU threshold (0.0-1.0, default: 0.50)
- **Max Detections**: Maximum detections per frame (default: 50)
- **Warmup Frames**: Frames to warm up model before measurement (default: 5)

#### Behavior Analysis

- **Enable Behavior Analysis**: Toggle real-time behavioral monitoring
- Metrics include: Activity (lethargy), Crowding, Edge-pacing, Inflow attraction

#### Heatmap Settings

- **Enable Heatmap**: Toggle occupancy heatmap visualization
- **Decay Rate**: Persistence of heatmap trails (0.50-0.999, higher = longer persistence)
- **Intensity**: Brightness multiplier for heatmap
- **Radius**: Detection circle radius on heatmap
- **Colormap**: Visual colormap (Jet, Inferno, Plasma, Magma, Turbo)

### Training Custom Models

#### 1. Prepare Training Data

Organize your dataset in YOLO format:

```
dataset/
├── data.yaml          # Dataset configuration
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/           # Instance segmentation masks
    ├── train/
    ├── val/
    └── test/
```

Example `data.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['sturgeon']
```

#### 2. Train with Optional Hyperparameter Tuning

Basic training:

```bash
python y11_seg_train.py --data dataset/data.yaml --epochs 100 --export-onnx
```

With hyperparameter tuning:

```bash
python y11_seg_train.py --data dataset/data.yaml --tune --tune-iterations 200 --export-onnx
```

Resume training:

```bash
python y11_seg_train.py --data dataset/data.yaml --resume runs_seg/y11seg_*/weights/last.pt
```

#### Training Options

```
--data DATA              Path to data.yaml (required)
--model MODEL            Base model: yolo11[n/s/m/l/x]-seg.pt (default: yolo11n-seg.pt)
--epochs EPOCHS          Training epochs (default: 100)
--imgsz IMGSZ           Image size (default: 640)
--batch BATCH           Batch size (-1 for auto)
--device DEVICE         GPU device: 0 or 0,1 or cpu
--workers WORKERS       DataLoader workers (default: auto)

--tune                  Enable hyperparameter tuning
--tune-iterations N     Tuning iterations (default: 300)
--tune-epochs N         Epochs per tuning iteration (default: 30)
--tune-workers N        DataLoader workers for tuning

--export-onnx          Export to ONNX format
--export-opset N       ONNX opset version (default: 13)
--export-dynamic       Enable dynamic axes in ONNX
--export-simplify      Simplify ONNX graph

--plots                Generate training plots
--seed SEED            Random seed (default: 42)
--exist-ok             Overwrite existing runs
```

#### Training Output

Training creates `runs_seg/<run_name>_artifacts/` with:

```
artifacts/
├── train/
│   ├── best.pt
│   ├── last.pt
│   ├── results.csv
│   ├── results.png
│   └── confusion_matrix.png
├── val/
│   ├── results.csv
│   └── plots/
├── test/
│   ├── results.csv
│   └── plots/
├── export/
│   └── best.onnx
├── training_process.json    # System info, args, metrics summary
├── train_events.jsonl       # Per-epoch metrics
├── run_summary.json         # High-level run summary
└── data.yaml               # Copy of training dataset config
```

## Project Structure

```
sturgeon-monitoring/
├── app.py                      # Main GUI application
├── y11_seg_train.py           # Training script with tuning
├── model-export.py            # Model export utility
├── pyproject.toml             # Project dependencies (uv)
├── requirements.txt           # pip dependencies
├── models/
│   ├── best.pt               # PyTorch model
│   ├── best.onnx             # ONNX model
│   ├── best.engine           # TensorRT model
│   ├── best.torchscript      # TorchScript model
│   └── best_openvino_model/  # OpenVINO model
├── inputs/                    # Input data directory
├── docs/                      # Documentation
│   ├── ultralytics-*.txt      # Ultralytics API references
│   └── openvino-*.txt         # OpenVINO integration guide
└── README.md
```

## Architecture

### GUI Components

**YoloTrackerWindow**: Main application window managing:

- Video frame acquisition and processing
- Real-time YOLO inference with streaming
- FPS metering and performance tracking
- Heatmap rendering and updates
- Behavior analysis coordination

**ConfigurationPanel**: Right-side control panel with:

- Model and source configuration
- Detection parameters
- Behavior analysis settings
- Heatmap controls

**BehaviorAnalyzer**: Background behavioral analysis:

- Tracks individual sturgeon movements
- Computes displacement, clustering, edge-pacing, and inflow metrics
- Generates z-score based alerts
- Handles night-time mode muting

**HeatmapManager**: Efficient heatmap visualization:

- O(1) Gaussian kernel caching
- Exponential decay rendering
- Customizable intensity and colormap

**BehaviorAnalysisThread**: Off-UI-thread analysis:

- Processes detection snapshots asynchronously
- Drops stale data to maintain responsiveness
- Configurable 10Hz analysis rate

### Performance Optimizations

- **Ring Buffer FPS Meter**: O(1) running sum with EMA smoothing
- **Streamed Inference**: Ultralytics `stream=True` for memory efficiency
- **Threaded Analysis**: Background behavior computation off UI thread
- **Kernel Caching**: Pre-computed Gaussian kernels for heatmap
- **Fast Luma Computation**: Downscaled frame brightness for night detection

## Virtual Camera Setup (OBS)

For monitoring via virtual camera:

### Windows

1. Install [OBS Studio](https://obsproject.com/)
2. Add your video source in OBS (Scene > Source)
3. Tools → VirtualCamera → Start
4. Run `app.py` and set Input Source to `0` (or enumerate via `cv2.VideoCapture`)

### Linux

```bash
# Install OBS with virtual camera support
sudo apt install obs-studio v4l2loopback-utils

# Check available cameras
ls /dev/video*
```

### macOS

```bash
# Install via homebrew
brew install --cask obs

# Set up virtual camera via OBS menu
# Tools → VirtualCamera → Start
```

## Monitoring Alerts

### Behavioral Thresholds

| Metric | Condition | Duration | Alert |
|--------|-----------|----------|-------|
| **Activity** | z-score ≤ -0.5 (lethargy) | 2s | "Low DO / post-handling / temperature shock?" |
| **Crowding** | z-score ≥ 3.0 (clustering) | 60s | "Potential water quality issue or disturbance" |
| **Edge-Pacing** | z-score ≥ 2.5 (wall-hugging) | 60s | "Stress / overcrowding / barren tank?" |
| **Inflow Magnet** | z-score ≥ 3.0 (ROI occupancy) | 60s | "Possible O₂ seeking / stratification / pump issue" |

Baselines are computed over 60-120 second windows. Night mode (luma < 28) disables activity alerts.

## Troubleshooting

### Model Loading Issues

**Error: "Model not found"**

- Ensure model file exists in `models/` directory
- Check file permissions
- Verify model format matches runtime (e.g., `.onnx` for ONNX runtime)

### Virtual Camera Not Detected

**OBS Studio:**

- Ensure "VirtualCamera" is started (Tools → VirtualCamera)
- Try camera index 1, 2, etc. if default is taken
- Restart OBS and app after changes

**Loopback device (Linux):**

```bash
# Load module if not loaded
sudo modprobe v4l2loopback

# List devices
v4l-utils list-devices
```

### Performance Issues

- Reduce image size: `--imgsz 416`
- Lower confidence threshold: `--confidence 0.40`
- Reduce max detections: `--max-detections 30`
- Use lighter model: `yolo11n-seg` instead of `yolo11x-seg`
- Disable behavior analysis: Uncheck "Enable Behavior Analysis"

### CUDA/GPU Issues

**CUDA not available:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**TensorRT installation fails:**

- Ensure NVIDIA CUDA 13.0 SDK is installed
- Install cuDNN matching CUDA version
- Fallback to ONNX or TorchScript formats

## Model Export

### PyTorch to ONNX

```bash
python y11_seg_train.py --data data.yaml --export-onnx
```

### PyTorch to TensorRT

Use `model-export.py` for TensorRT export:

```bash
python model-export.py --model models/best.pt --format engine --export-path models/best.engine
```

### PyTorch to OpenVINO

```bash
ovc models/best.pt --output-model models/best_openvino_model
```

## Logging

Logs are written to `yolo_tracker.log` in the application directory:

```plaintext
2025-11-09 10:30:15,123 - __main__ - INFO - Loading YOLO model for segmentation task...
2025-11-09 10:30:18,456 - __main__ - INFO - Model loaded successfully
2025-11-09 10:30:19,789 - __main__ - WARNING - ALERT Activity: Low DO / post-handling / temperature shock?
```

## API Reference

### Running Custom Detection

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt', task='segment')

# Stream inference
results = model.predict(
    source='rtsp://camera.local/stream',
    conf=0.50,
    iou=0.50,
    max_det=50,
    stream=True
)

for result in results:
    boxes = result.boxes.xyxy        # [N, 4] coordinates
    ids = result.boxes.id            # [N] tracking IDs
    masks = result.masks.xy          # [N, M, 2] polygon points
    print(f"Detected {len(boxes)} sturgeons")
```

### Custom Behavior Analysis

```python
from app import BehaviorAnalyzer, BehaviorMeta, BoxesSnapshot
import numpy as np

analyzer = BehaviorAnalyzer()

# Prepare snapshot
meta = BehaviorMeta(frame_wh=(640, 480), avg_luma=100.0)
snap = BoxesSnapshot(
    xyxy=np.array([[100, 100, 200, 200], ...]),
    ids=np.array([1, 2, ...]),
    conf=np.array([0.95, 0.92, ...])
)

# Analyze
result = analyzer.analyze(meta, snap)
for key, reading in result.readings.items():
    print(f"{reading.title}: {reading.value_text}")
```

## References

- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Object Tracking](https://docs.ultralytics.com/modes/track/)
- [OpenVINO Deployment](https://docs.openvino.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- GUI powered by [PySide6](https://doc.qt.io/qtforpython/)
- Advanced analytics with [NumPy](https://numpy.org/) and [OpenCV](https://opencv.org/)

---

**Questions or Issues?** Please open an issue or contact the project maintainers.
