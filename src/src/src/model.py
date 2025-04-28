from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """Make predictions using the trained model."""
    return model.predict(X_test)
