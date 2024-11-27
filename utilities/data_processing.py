# data_processing.py

import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocess the data by converting timestamps and sorting.
    """
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert timestamp columns to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)

    # Sort the DataFrame by timestamp
    df.sort_values('timestamp', inplace=True)

    # Reset index if necessary
    df.reset_index(drop=True, inplace=True)

    return df

def create_features(df):
    """
    Create features and the target variable.
    """
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create target variable 'direction'
    df['future_funding_rate'] = df['funding_rate'].shift(-1)
    df['direction'] = (df['future_funding_rate'] > df['funding_rate']).astype(int)
    df.dropna(subset=['future_funding_rate'], inplace=True)

    return df
