import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# ===========================================
# Feature Engineering Functions
# ===========================================

def add_lag_features(df):
    """
    Add lag features to capture temporal trends.
    """
    if 'funding_rate' not in df.columns:
        raise KeyError("'funding_rate' column is missing. Check the input data.")

    # Lagged funding rate features
    df['funding_rate_lag1'] = df['funding_rate'].shift(1)
    df['funding_rate_lag2'] = df['funding_rate'].shift(2)

    if 'open_interest' in df.columns:
        df['open_interest_lag1'] = df['open_interest'].shift(1)
    else:
        df['open_interest_lag1'] = np.nan

    if 'mark_price' in df.columns:
        df['mark_price_lag1'] = df['mark_price'].shift(1)
    else:
        df['mark_price_lag1'] = np.nan

    # Handle missing values from lagging
    df.fillna(0, inplace=True)
    return df

def add_technical_indicators(df):
    """
    Add technical indicators like moving averages to the DataFrame.
    """
    if 'funding_rate' not in df.columns:
        raise KeyError("'funding_rate' column is missing in the DataFrame. Check input data.")
    
    df['funding_rate_ma3'] = df['funding_rate'].rolling(window=3).mean()
    df['funding_rate_ma5'] = df['funding_rate'].rolling(window=5).mean()

    return df

def add_interaction_terms(df):
    """
    Add interaction terms to capture relationships between features.
    """
    if 'funding_rate_lag1' not in df.columns or 'funding_rate_lag2' not in df.columns:
        raise KeyError("'funding_rate_lag1' or 'funding_rate_lag2' columns are missing. Ensure lag features are added first.")
    if 'funding_rate_ma3' not in df.columns:
        raise KeyError("'funding_rate_ma3' column is missing. Ensure technical indicators are added first.")

    df['interaction1'] = df['funding_rate_lag1'] * df['funding_rate_lag2']

    # Handle potential division-by-zero issues for interaction2
    interaction2 = df['funding_rate_ma3'] / (df['funding_rate_lag1'].replace(0, np.nan) + 1e-6)
    interaction2 = interaction2.replace([np.inf, -np.inf], np.nan)
    interaction2 = interaction2.fillna(0)

    # Assign the processed interaction2 back to the DataFrame
    df['interaction2'] = interaction2

    return df

# ===========================================
# Data Sampling and Balancing
# ===========================================

def apply_smote(X_train, y_train, sampling_strategy=1.0, random_state=42):
    """
    Apply SMOTE to balance the classes in the training data.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        sampling_strategy (float or dict): Sampling strategy for SMOTE.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_resampled, y_resampled: Balanced training data.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled