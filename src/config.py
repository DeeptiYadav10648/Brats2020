from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

PATCH_SIZE = (8,8,8)

BATCH_SIZE = 1

NUM_WORKERS = 4

VAL_SPLIT = 0.2

SEED = 42