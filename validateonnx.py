import onnx
import onnxruntime as ort
import numpy as np

def validate_onnx_model(model_path):
    try:
        # Load the ONNX model
        print(f"[INFO] Validating ONNX model: {model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("[SUCCESS] ONNX model structure is VALID ✅")

        # Inference Test (with dummy data)
        session = ort.InferenceSession(model_path)
        inputs = session.get_inputs()
        input_shape = inputs[0].shape
        input_name = inputs[0].name

        print(f"[INFO] Model Input Name: {input_name}")
        print(f"[INFO] Model Input Shape: {input_shape}")

        # Replace dynamic dimensions with 1
        fixed_input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]

        # Dummy Input (Random Noise for Testing)
        dummy_input = np.random.rand(*fixed_input_shape).astype(np.float32)
        result = session.run(None, {input_name: dummy_input})
        print("[SUCCESS] ONNX Model Inference Passed ✅")

    except Exception as e:
        print(f"[ERROR] ONNX Model Validation Failed 🚨: {e}")

if __name__ == "__main__":
    validate_onnx_model("model.onnx")
