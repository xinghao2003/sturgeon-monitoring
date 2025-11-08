from ultralytics import YOLO

# Load model
model = YOLO(r"C:\Users\xingh\Desktop\sturgeon-monitoring-v2\models\best.pt")

try:
    # model.export(format="openvino")
    # model.export(format="engine")
    # model.export(format="onnx")
    model.export(format="torchscript")
    # model.export(format="tflite")
    # model.export(format="edgetpu")
    # model.export(format="executorch")
except Exception as e:
    print(f"Export failed: {e}")
    import traceback
    traceback.print_exc()