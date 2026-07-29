import pandas as pd
from sklearn.model_selection import train_test_split
from utils.model_loader import load_preprocessor

def preprocess(df):
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    
    preprocessor = load_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit & Transform train, Transform test
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor