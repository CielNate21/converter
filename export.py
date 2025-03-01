import sys
import os
import torch
import onnx
import tf2onnx
import tensorflow as tf
import numpy as np

# 1. Automatically Add Yolov9 Folder to Python Path
sys.path.insert(0, os.path.dirname(r"C:\Users\Nathaniel de Guzman\PycharmProjects\yolov9-main"))

# 2. Get the Folder Where export.py is Running 🔥
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. Load YOLOv9 Custom Model
try:
    model_dict = torch.load(os.path.join(CURRENT_DIR, "models", "weights.pt"), map_location='cpu', weights_only=False)
    model = model_dict['model']
    model.eval()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Failed:", e)
    exit()

# 4. Dummy Input (for ONNX Export)
dummy_input = torch.randn(1, 3, 640, 640)

# 5. Export to ONNX
onnx_output_path = os.path.join(CURRENT_DIR, "model.onnx")
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
    from onnx_tf.backend import prepare

    # Load ONNX model
    model_onnx = onnx.load(onnx_output_path)

    # Convert ONNX to TensorFlow
    tf_rep = prepare(model_onnx)

    # Save TensorFlow model inside the same folder
    tf_saved_model_dir = os.path.join(CURRENT_DIR, "model_saved")
    tf_rep.export_graph(tf_saved_model_dir)
    print(f"✅ TensorFlow SavedModel Exported Successfully at: {tf_saved_model_dir}")

except Exception as e:
    print("❌ TensorFlow Export Failed:", e)
    exit()

print("🎯 PHASE 2: TensorFlow Conversion Completed")

# 7. Convert TensorFlow SavedModel to TFLite 🚨 FINAL BOSS
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    # FP32
    tflite_model_fp32 = converter.convert()
    with open(os.path.join(CURRENT_DIR, "model_fp32.tflite"), "wb") as f:
        f.write(tflite_model_fp32)
    print("🔥 TFLite FP32 Model Exported Successfully")

    # FP16
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_fp16 = converter.convert()
    with open(os.path.join(CURRENT_DIR, "model_fp16.tflite"), "wb") as f:
        f.write(tflite_model_fp16)
    print("🔥 TFLite FP16 Model Exported Successfully")

    # INT8 (The Android Final Boss 💀)
    def representative_dataset():
        for _ in range(100):
            dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
            yield [dummy_input]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_int8 = converter.convert()
    with open(os.path.join(CURRENT_DIR, "model_int8.tflite"), "wb") as f:
        f.write(tflite_model_int8)
    print("🔥 TFLite INT8 Model Exported Successfully (Recommended for Android)")

except Exception as e:
    print("❌ TFLite Conversion Failed:", e)
    exit()

print("🎯 PHASE 3: TFLite Conversion Completed")
