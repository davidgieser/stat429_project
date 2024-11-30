import pandas as pd
import numpy as np
import os

from utilities.functions import (
    add_lag_features,
    add_technical_indicators,
    add_interaction_terms,
    # add_rate_of_change,
    # add_advanced_features,
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

def preprocess_data(df, handle_timestamps=True):
    """
    Preprocess the data by handling timestamps, sorting, and cleaning missing or infinite values.
    """
    df = df.copy()

    if handle_timestamps and 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df.sort_values('timestamp', inplace=True)
        except Exception as e:
            print(f"Error during timestamp processing: {e}")
    elif handle_timestamps:
        print("'timestamp' column is missing. Skipping timestamp processing.")

    # Handle infinite and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Reset index after processing
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
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(df.head())  # Preview the initial data
    except Exception as e:
        print(f"Error during dataset loading: {e}")
        return None
    
    try:
        print("Checking for 'timestamp' column...")
        if 'timestamp' not in df.columns:
            print("'timestamp' column is missing. Adding dummy timestamps.")
            df['timestamp'] = pd.Timestamp.now()

        print("Preprocessing the data...")
        df = preprocess_data(df)
        print(f"Data preprocessing completed. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None

    # List of pipeline steps
    pipeline_steps = [
        ("Preprocessing the data", preprocess_data),
        ("Adding lag features", add_lag_features),
        ("Adding technical indicators", add_technical_indicators),
        ("Adding interaction terms", add_interaction_terms),
        ("Creating features and target variable", create_features),
    ]

    for step_name, step_function in pipeline_steps:
        try:
            print(f"{step_name}...")
            df = step_function(df)
            print(f"{step_name} completed. Shape: {df.shape}")
            print(df.head())  # Optional: Preview data after each step
        except Exception as e:
            print(f"Error during {step_name.lower()}: {e}")
            return None

    print("Pipeline completed successfully.")
    return df