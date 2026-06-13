"""
Configuration for IDM Backend
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if not exists
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
CLUSTERING_MODEL_PATH = MODELS_DIR / 'kmeans_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler_features.pkl'
CLASSIFIER_PATH = MODELS_DIR / 'classifier_model.pkl'
LABEL_ENCODER_PATH = MODELS_DIR / 'label_encoder.pkl'

# Data file paths
RAW_DATA_PATH = DATA_DIR / 'idm_2024_raw.csv'
CLEAN_DATA_PATH = DATA_DIR / 'idm_2024_clean.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'idm_2024_processed.csv'

# Feature columns
INDEX_COLS = ['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']
FEATURE_COLS = ['IKS_2024', 'IKE_2024', 'IKL_2024']
TARGET_COL = 'STATUS_IDM_2024'

# Status mapping
STATUS_MAP = {
    'Sangat Tertinggal': 1,
    'Tertinggal': 2,
    'Berkembang': 3,
    'Maju': 4,
    'Mandiri': 5
}

# Clustering parameters
KMEANS_K = 3
KMEANS_RANDOM_STATE = 42

# Model parameters
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
