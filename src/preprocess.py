import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data("C:\Users\User\Downloads\creditcard.csv (1).zip"):
    """Load dataset from a CSV file."""
    return pd.read_csv("C:\Users\User\Downloads\creditcard.csv (1).zip")

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    return df.fillna(df.median())

def split_data(df, target_col):
    """Split the dataset into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def balance_classes(X, y):
    """Handle class imbalance using SMOTE."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)
