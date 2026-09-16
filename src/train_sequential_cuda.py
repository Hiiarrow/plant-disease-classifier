import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train_unified import train

class Args:
    def __init__(self, model_name, epochs=8, batch_size=32):
        self.crop = "all"
        self.data_dir = None
        self.model = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = 5e-4
        self.val_ratio = 0.2

def main():
    # mobilenet_v3 is already trained & saved at 99.84% accuracy. Train remaining models:
    models_to_train = ["attention_mobilenet", "resnet50"]
    print("==================================================")
    print(" LAUNCHING HIGH-PRECISION CUDA GPU TRAINING (SEQUENTIAL)")
    print("==================================================")
    
    for m in models_to_train:
        print(f"\n>>> Starting CUDA Training for Model: {m.upper()} <<<")
        args = Args(model_name=m, epochs=8, batch_size=32)
        train(args)
        print(f"[DONE] Completed CUDA Training for Model: {m.upper()}")

if __name__ == "__main__":
    main()
