from pathlib import Path
import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parents[1] / "models" / "house_price_model.joblib"

def predict_price(property_data: dict):
    model = joblib.load(MODEL)
    prediction = float(model.predict(pd.DataFrame([property_data]))[0])
    return prediction
