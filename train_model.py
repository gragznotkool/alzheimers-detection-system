# src/pipeline/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    # Load cleaned data
    df = pd.read_csv("data/processed/alzheimers_cleaned.csv")

    # Split features and target
    X = df.drop("target", axis=1)    # replace "target" with your column name
    y = df["target"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model training complete. Model saved to artifacts/model.pkl")

if __name__ == "__main__":
    train_model()
