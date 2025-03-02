import os

def validate_tflite_headers():
    tflite_files = ["model_fp32.tflite", "model_fp16.tflite", "model_int8.tflite"]
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    for tflite_file in tflite_files:
        file_path = os.path.join(CURRENT_DIR, tflite_file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"[INFO] File '{tflite_file}' exists ✅")
            print(f"[INFO] File Size: {file_size / (1024 * 1024):.2f} MB")

            with open(file_path, "rb") as f:
                header = f.read(8)
                if b"TFL3" in header:
                    print(f"[SUCCESS] {tflite_file} Header Found ✅")
                else:
                    print(f"[ERROR] {tflite_file} Corrupted ❌")
        else:
            print(f"[ERROR] {tflite_file} Not Found ❌")

if __name__ == "__main__":
    validate_tflite_headers()