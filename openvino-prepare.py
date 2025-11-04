from ultralytics import YOLO

# Export PyTorch → OpenVINO
YOLO(r"C:\Users\xingh\Desktop\sturgeon-monitoring\runs_seg\y11seg_20251103-041452\weights\best.pt").export(format="openvino")  # creates yolo11n_openvino_model/