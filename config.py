import os

# ===========================================
# Directory Configurations
# ===========================================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'normalized_data')

# Path to the main dataset
BINANCE_BTC_PERP_CSV = os.path.join(BASE_DIR, 'data', 'normalized_datasets', 'binance_btc_perp.csv')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Model file paths
MODEL1_PATH = os.path.join(SAVED_MODELS_DIR, 'model1.pkl')
SCALER1_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler1.pkl')
MODEL1_LOGREG_PATH = os.path.join(SAVED_MODELS_DIR, 'model1_logistic_regression.pkl')
SCALER1_LOGREG_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler1_logistic_regression.pkl')
MODEL2_PATH = os.path.join(SAVED_MODELS_DIR, 'model2_garch.pkl')

# Instrument specifications
TARDIS_TYPE = 'derivative_ticker'
INSTRUMENT = 'btcusdt'
START_DATE = '2020-01-01'
END_DATE = '2024-10-18'
EXCHANGE = 'binance-futures'

# ===========================================
# General Configurations
# ===========================================

# Random seed for reproducibility
RANDOM_STATE = 42