import pandas as pd
import numpy as np
import os

from utilities.functions import (
    add_lag_features,
    add_technical_indicators,
    add_interaction_terms,
)

# ===========================================
# Data Loading and Preprocessing
# ===========================================

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded. Preview:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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

# ===========================================
# Feature Engineering
# ===========================================
def create_features(df):
    """
    Create additional features and the target variable.
    """
    # Time-based and cyclical features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create target variable 'direction'
    df['future_funding_rate'] = df['funding_rate'].shift(-1)
    df['direction'] = (df['future_funding_rate'] > df['funding_rate']).astype(int)
    df.drop(columns=['future_funding_rate'], inplace=True)

    return df

# ===========================================
# Processing Pipeline
# ===========================================
def process_pipeline(filepath):
    """
    Complete processing pipeline for loading, preprocessing, and feature creation.
    """
    try:
        print("Loading the dataset...")
        df = load_data(filepath)
        if df.empty:
            raise ValueError("Dataset is empty. Check the input file.")
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during dataset loading: {e}")
        return None

    try:
        print("Preprocessing the data...")
        df = preprocess_data(df)
        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing.")
        print(f"Data preprocessing completed. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None

    try:
        print("Adding lag features...")
        df = add_lag_features(df)
        if 'funding_rate_lag1' not in df.columns:
            raise KeyError("'funding_rate_lag1' not created. Check `add_lag_features`.")
        print(f"Lag features added. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during adding lag features: {e}")
        return None

    try:
        print("Adding technical indicators...")
        df = add_technical_indicators(df)
        print(f"Technical indicators added. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during adding technical indicators: {e}")
        return None

    try:
        print("Adding interaction terms...")
        df = add_interaction_terms(df)
        print(f"Interaction terms added. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during adding interaction terms: {e}")
        return None

    try:
        print("Creating features and target variable...")
        df = create_features(df)
        print(f"Features and target variable created. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during creating features and target variable: {e}")
        return None

    try:
        print("Filling missing values with 0...")
        df.fillna(0, inplace=True)
        print("Missing values filled.")
    except Exception as e:
        print(f"Error during filling missing values: {e}")
        return None

    try:
        print("Dropping rows with missing values...")
        df.dropna(inplace=True)
        print(f"Rows with missing values dropped. Final shape: {df.shape}")
    except Exception as e:
        print(f"Error during dropping rows with missing values: {e}")
        return None

    return df