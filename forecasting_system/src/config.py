import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
HISTORY_DIR = MODELS_DIR / "history"

# Data settings
TARGET_COL = "Total"
DATE_COL = "Date"
GROUP_COLS = ["State", "Category"]
FORECAST_HORIZON = 3 # months

# Feature engineering settings
LAG_FEATURES = [1, 3, 6] # months to lag
ROLLING_WINDOWS = [3, 6] # months for rolling stats

# Model settings
TEST_SIZE_MONTHS = 3 # Number of months to hold out for validation

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BEST_MODEL_DIR, HISTORY_DIR]:
    os.makedirs(directory, exist_ok=True)
