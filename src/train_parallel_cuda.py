import os
import subprocess
import sys
import time

def run_parallel_cuda_training(epochs=8, batch_size=16):
    print("==================================================")
    print(" LAUNCHING PARALLEL CUDA GPU TRAINING FOR ALL 3 MODELS")
    print("==================================================")

    python_exe = sys.executable
    models = ["mobilenet_v3", "attention_mobilenet", "resnet50"]
    processes = []

    for model_name in models:
        cmd = [
            python_exe, "-u", "-m", "src.train_unified",
            "--crop", "all",
            "--model", model_name,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size)
        ]
        print(f"[*] Launching CUDA Process for model: {model_name}...")
        p = subprocess.Popen(cmd)
        processes.append((model_name, p))
        time.sleep(2)  # Short delay between process initialization

    print("\n[INFO] All 3 CUDA processes are now executing concurrently on GPU!")
    print("[INFO] Waiting for all 3 parallel training runs to complete...\n")

    for model_name, p in processes:
        p.wait()
        print(f"✓ Parallel CUDA Training finished for: {model_name} (Exit code: {p.returncode})")

    print("\n[SUCCESS] Parallel CUDA GPU Training complete for all 3 models!")

if __name__ == "__main__":
    run_parallel_cuda_training()
