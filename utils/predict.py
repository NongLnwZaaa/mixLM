import pandas as pd

def predict(model, preprocessor, data):

    df = pd.DataFrame([data])

    X = preprocessor.transform(df)

    pred = model.predict(X)

    prob = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)

    return pred, prob