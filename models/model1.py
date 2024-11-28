# Import necessary modules and utilities
import pandas as pd
import numpy as np
from utilities.data_processing import load_data, preprocess_data, create_features
from utilities.model_utils import train_classification_model, evaluate_classification_model, save_model
from config import (
    BINANCE_BTC_PERP_CSV,
    MODEL1_LOGREG_PATH,
    SCALER1_LOGREG_PATH,
    RANDOM_STATE
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def train_model1_logistic_regression(data_filepath, model_filepath, scaler_filepath):
    """
    Train Model 1 using Logistic Regression to predict the direction of funding rate movement.
    """
    # Load and preprocess data
    df = load_data(data_filepath)
    df = preprocess_data(df)
    df = create_features(df)

    # Create the 'direction' target variable
    df['future_funding_rate'] = df['funding_rate'].shift(-1)
    df.dropna(subset=['future_funding_rate'], inplace=True)
    df['direction'] = (df['future_funding_rate'] > df['funding_rate']).astype(int)
    df.drop(columns=['future_funding_rate'], inplace=True)

    # Define features and target
    feature_columns = [
        'open_interest', 'mark_price',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    X = df[feature_columns]
    y = df['direction']

    # Handle missing values if any
    X = X.dropna()
    y = y.loc[X.index]

    # Split data into training and testing sets without shuffling (time series)
    split_index = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['open_interest', 'mark_price']])
    X_test_scaled = scaler.transform(X_test[['open_interest', 'mark_price']])

    # Convert scaled features back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=['open_interest', 'mark_price'], index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['open_interest', 'mark_price'], index=X_test.index)

    # Combine scaled numerical features with cyclical features
    X_train_prepared = pd.concat(
        [X_train_scaled_df.reset_index(drop=True), X_train[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']].reset_index(drop=True)],
        axis=1
    )
    X_test_prepared = pd.concat(
        [X_test_scaled_df.reset_index(drop=True), X_test[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']].reset_index(drop=True)],
        axis=1
    )

    # Initialize and train the Logistic Regression model
    logreg_model = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    )
    logreg_model = train_classification_model(logreg_model, X_train_prepared, y_train)

    # Evaluate the model
    evaluate_classification_model(logreg_model, X_test_prepared, y_test)

    # Save the trained model and scaler
    save_model(logreg_model, model_filepath)
    save_model(scaler, scaler_filepath)

if __name__ == "__main__":
    # Define file paths using config.py
    data_filepath = BINANCE_BTC_PERP_CSV
    model_filepath = MODEL1_LOGREG_PATH
    scaler_filepath = SCALER1_LOGREG_PATH

    # Train Model 1 using Logistic Regression
    train_model1_logistic_regression(data_filepath, model_filepath, scaler_filepath)