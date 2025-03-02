import tensorflow as tf
import numpy as np
import os

def validate_saved_model(saved_model_dir):
    try:
        print(f"[INFO] Validating TensorFlow SavedModel: {saved_model_dir}")

        # Load the model
        model = tf.saved_model.load(saved_model_dir)
        print("[SUCCESS] TensorFlow SavedModel is VALID ✅")

        # Check Model Signatures
        print("[INFO] Model Signatures:")
        print(model.signatures)

        # Run Inference Test
        infer = model.signatures["serving_default"]

        input_tensor = list(infer.structured_input_signature[1].values())[0]
        input_shape = input_tensor.shape
        input_dtype = input_tensor.dtype

        print(f"[INFO] Model Input Shape: {input_shape}")
        print(f"[INFO] Model Input Type: {input_dtype}")

        # Dummy Input Test
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        result = infer(tf.constant(dummy_input))

        print("[SUCCESS] TensorFlow Model Inference Passed ✅")

    except Exception as e:
        print(f"[ERROR] TensorFlow Model Validation Failed 🚨: {e}")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    validate_saved_model(os.path.join(CURRENT_DIR, "model_saved"))
    