# utilities/functions.py

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def add_lag_features(df):
    """
    Add lag features to the DataFrame.
    """
    df['funding_rate_lag1'] = df['funding_rate'].shift(1)
    df['funding_rate_lag2'] = df['funding_rate'].shift(2)  # Uncommented
    df['open_interest_lag1'] = df['open_interest'].shift(1)
    df['mark_price_lag1'] = df['mark_price'].shift(1)
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    df['funding_rate_ma3'] = df['funding_rate'].rolling(window=3).mean()
    df['funding_rate_ma5'] = df['funding_rate'].rolling(window=5).mean()  # Uncommented
    return df

def apply_smote(X_train_prepared, y_train, random_state=42):
    """
    Apply SMOTE to balance the classes in the training data.
    """
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prepared, y_train)
    return X_train_resampled, y_train_resampled

def perform_hyperparameter_tuning(X_train, y_train, random_state=42):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_classification_model(model, X_test, y_test, y_proba=None, threshold=0.5):
    """
    Evaluate the classification model's performance.
    """
    if y_proba is None:
        y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance from the model.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def train_classification_model(model, X_train, y_train):
    """
    Train the classification model.
    """
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a model from a file.
    """
    import joblib
    return joblib.load(filepath)
