import sys
import os
import torch
import onnx
import tensorflow as tf
import numpy as np
import importlib
from onnx_tf.backend import prepare

# 1. Automatically Add Yolov9 Folder to Python Path
yolov9_path = os.path.join(os.getcwd(), "yolov9-main")
if yolov9_path not in sys.path:
    sys.path.insert(0, yolov9_path)
    print("🔄 Python Path Reloaded for yolov9 Folder")

# Clean Cache
importlib.invalidate_caches()
import models.experimental
print("✅ Cache Cleared and Reloaded!")

# 2. Verify Yolov9 Folder
if not os.path.exists(os.path.join(yolov9_path, "models")):
    print("❌ yolov9 models folder not found!")
    exit()
else:
    print("✅ yolov9 Folder Added to Python Path")

try:
    from models.experimental import attempt_load
    print("🚀 Import Successful: attempt_load()")
except ModuleNotFoundError as e:
    print("❌ Import Failed:", e)
    exit()

# 3. Load YOLOv9 Custom Model
try:
    best_pt_path = os.path.join(os.getcwd(), "Custom Acne Yolov9 Model", "train", "exp", "weights", "best.pt")
    model = attempt_load(best_pt_path)
    model.eval()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Failed:", e)
    exit()

# 4. Dummy Input (for ONNX Export)
dummy_input = torch.randn(1, 3, 640, 640)

# 5. Export to ONNX
onnx_output_path = os.path.join(os.getcwd(), "model.onnx")
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output']
    )
    print("🚀 ONNX Model Exported Successfully")
except Exception as e:
    print("❌ ONNX Export Failed:", e)
    exit()

print("🎯 PHASE 1: ONNX Export Completed")

# 6. Convert ONNX to TensorFlow SavedModel using onnx-tf
try:
    # Load ONNX model
    model_onnx = onnx.load(onnx_output_path)

    # Convert ONNX to TensorFlow
    tf_rep = prepare(model_onnx)

    # Save TensorFlow model
    tf_saved_model_dir = os.path.join(os.getcwd(), "model_saved")
    tf_rep.export_graph(tf_saved_model_dir)
    print(f"✅ TensorFlow SavedModel Exported Successfully at: {tf_saved_model_dir}")

except Exception as e:
    print("❌ TensorFlow Export Failed:", e)
    exit()

print("🎯 PHASE 2: TensorFlow Conversion Completed")

# 7. Convert TensorFlow SavedModel to TFLite
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    # FP32 Conversion
    tflite_model_fp32 = converter.convert()
    with open("model_fp32.tflite", "wb") as f:
        f.write(tflite_model_fp32)
    print("🔥 TFLite FP32 Model Exported Successfully")

    # FP16 Conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_fp16 = converter.convert()
    with open("model_fp16.tflite", "wb") as f:
        f.write(tflite_model_fp16)
    print("🔥 TFLite FP16 Model Exported Successfully")

    # INT8 Conversion
    def representative_dataset():
        for _ in range(100):
            yield [np.random.rand(1, 3, 640, 640).astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_int8 = converter.convert()
    with open("model_int8.tflite", "wb") as f:
        f.write(tflite_model_int8)
    print("🔥 TFLite INT8 Model Exported Successfully")

except Exception as e:
    print("❌ TFLite Conversion Failed:", e)
    exit()

print("🎯 PHASE 3: TFLite Conversion Completed")
