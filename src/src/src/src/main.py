import pandas as pd
from src.preprocess import load_data, handle_missing_values, split_data, balance_classes
from src.feature_engineering import create_features
from src.model import train_model, predict
from src.evaluate import evaluate_model

# Load and preprocess data
data = load_data('C:\Users\User\Downloads\creditcard.csv (1).zip')
data = handle_missing_values(data)
data = create_features(data)

# Split and balance data
X_train, X_test, y_train, y_test = split_data(data, target_col='is_fraud')
X_train, y_train = balance_classes(X_train, y_train)

# Train and evaluate the model
model = train_model(X_train, y_train)
y_pred = predict(model, X_test)
metrics = evaluate_model(y_test, y_pred)

# Display evaluation metrics
print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")
