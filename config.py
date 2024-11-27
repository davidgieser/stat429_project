# config.py

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'normalized_datasets')

# Path to the main dataset
BINANCE_BTC_PERP_CSV = os.path.join(NORMALIZED_DATA_DIR, 'binance_btc_perp.csv')

# Tardis API Configuration
# Do not fetch TARDIS_API_KEY here to avoid import issues
# TARDIS_API_KEY will be fetched when needed in data_pull.py

TARDIS_TYPE = 'derivative_ticker'
QUOTE = 'BTC'
BASE = 'USDT'
START_DATE = '2020-01-01'
END_DATE = '2024-10-19'
EXCHANGE = 'binance-futures'  # Ensure this line is present

# Instrument
INSTRUMENT = f"{QUOTE.lower()}{BASE.lower()}"

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Model file paths
MODEL1_PATH = os.path.join(SAVED_MODELS_DIR, 'model1.pkl')
SCALER1_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler1.pkl')

# Random seed for reproducibility
RANDOM_STATE = 42

# Test size for train-test split
TEST_SIZE = 0.2