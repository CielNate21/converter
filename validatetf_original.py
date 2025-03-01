import tensorflow as tf
import numpy as np
import os

def validate_tflite_model(model_path, precision):
    try:
        print(f"[INFO] Validating TFLite Model: {model_path} ({precision})")

        # Load TFLite Model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Model Input Details
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details['shape']
        input_dtype = input_details['dtype']

        print(f"[INFO] Model Input Shape: {input_shape}")
        print(f"[INFO] Model Input Type: {input_dtype}")

        # Model Output Details
        output_details = interpreter.get_output_details()[0]
        output_shape = output_details['shape']
        output_dtype = output_details['dtype']

        print(f"[INFO] Model Output Shape: {output_shape}")
        print(f"[INFO] Model Output Type: {output_dtype}")

        # Dummy Input
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

        # Quantization Check for INT8 💀
        if precision == "INT8":
            dummy_input = (dummy_input * 255).astype(np.int8)  # Simulate INT8 Quantization
            interpreter.set_tensor(input_details['index'], dummy_input)
        else:
            interpreter.set_tensor(input_details['index'], dummy_input)

        interpreter.invoke()

        result = interpreter.get_tensor(output_details['index'])

        print(f"[SUCCESS] TFLite Model Inference Passed ✅ ({precision})")

        # INT8 Output Validation
        if precision == "INT8" and result.dtype != np.int8:
            print("[ERROR] INT8 Model Output is NOT Quantized 🚨")
            exit()

    except Exception as e:
        print(f"[ERROR] TFLite Model Validation Failed 🚨: {e}")
        exit()

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Validate FP32
    validate_tflite_model(os.path.join(CURRENT_DIR, "model_fp32.tflite"), "FP32")

    # Validate FP16
    validate_tflite_model(os.path.join(CURRENT_DIR, "model_fp16.tflite"), "FP16")

    # Validate INT8
    validate_tflite_model(os.path.join(CURRENT_DIR, "model_int8.tflite"), "INT8")

    print("🎯 All TFLite Models Validation Completed ✅")
