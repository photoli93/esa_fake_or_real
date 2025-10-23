import os
from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# Text processing
MAX_SEQ_LENGTH = 512
MIN_SEQ_LENGTH = 10

# Tokenizer
TOKENIZER_NAME = "distilbert-base-uncased"
VOCAB_SIZE = 30522
