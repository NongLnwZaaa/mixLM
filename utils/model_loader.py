import joblib

def load_model(name):
    return joblib.load(f"models/{name}")

def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")