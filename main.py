from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(title="AQI Prediction API")

# Load model and supporting files
model = joblib.load("model/aqi_model_india.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")
state_enc_map = joblib.load("model/state_enc_map.pkl")


class AQIInput(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    ozone: float
    rh: float
    ws: float
    sr: float
    state: str
    pm25_lag1: float
    aqi_lag1: float
    date: str  # format: YYYY-MM-DD


def get_season(month):
    if month in [12, 1, 2]: return 0
    if month in [3, 4, 5]: return 1
    if month in [6, 7, 8]: return 2
    return 3


@app.get("/")
def root():
    return {"message": "AQI Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_aqi(data: AQIInput):
    date_obj = datetime.strptime(data.date, "%Y-%m-%d")
    month = date_obj.month
    day_of_year = date_obj.timetuple().tm_yday
    season = get_season(month)
    state_enc = state_enc_map.get(data.state, -1)

    row = pd.DataFrame([{
        "PM2.5 (ug/m3)": data.pm25,
        "PM10 (ug/m3)": data.pm10,
        "NO2 (ug/m3)": data.no2,
        "SO2 (ug/m3)": data.so2,
        "CO (mg/m3)": data.co,
        "Ozone (ug/m3)": data.ozone,
        "RH (%)": data.rh,
        "WS (m/s)": data.ws,
        "SR (W/mt2)": data.sr,
        "month": month,
        "day_of_year": day_of_year,
        "season": season,
        "state_enc": state_enc,
        "PM2.5_lag1": data.pm25_lag1,
        "AQI_lag1": data.aqi_lag1,
    }])[feature_cols]

    prediction = model.predict(row)[0]

    return {
        "predicted_aqi": round(float(prediction), 1),
        "category": get_aqi_category(prediction)
    }


def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"