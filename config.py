from pathlib import Path

import torch

DATASET_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

VERSION = "b0"
NUM_CLASSES = 90
IMG_SIZE = 224

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
