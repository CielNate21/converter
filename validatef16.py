import os

tflite_file = r"C:\Users\Nathaniel de Guzman\PycharmProjects\newThesis\model_fp16.tflite"

if os.path.exists(tflite_file):
    file_size = os.path.getsize(tflite_file)
    print(f"[INFO] File '{tflite_file}' exists ✅")
    print(f"[INFO] File Size: {file_size / (1024 * 1024):.2f} MB")

    with open(tflite_file, "rb") as f:
        header = f.read(8)
        if b"TFL3" in header:
            print("[SUCCESS] TFLite Model Header Found ✅")
        else:
            print("[ERROR] TFLite Model Corrupted ❌")
else:
    print("[ERROR] File Not Found ❌")
