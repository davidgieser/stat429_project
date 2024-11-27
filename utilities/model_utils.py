import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

def train_classification_model(model, X_train, y_train):
    """
    Train a classification model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the classification model's performance.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a model from a file.
    """
    return joblib.load(filepath)
