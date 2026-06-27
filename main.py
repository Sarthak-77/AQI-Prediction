from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(title="AQI Prediction API")

# Allow browser clients (the React frontend) to call the API.
# Tighten allow_origins to your deployed frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and supporting files
model = joblib.load("model/aqi_model_india.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")
state_enc_map = joblib.load("model/state_enc_map.pkl")


class AQIInput(BaseModel):
    # Pollutant concentrations — non-negative, with generous upper bounds that
    # comfortably cover real-world extremes while rejecting nonsense input.
    pm25: float = Field(ge=0, le=1000, description="PM2.5 (µg/m³)")
    pm10: float = Field(ge=0, le=1000, description="PM10 (µg/m³)")
    no2: float = Field(ge=0, le=1000, description="NO₂ (µg/m³)")
    so2: float = Field(ge=0, le=1000, description="SO₂ (µg/m³)")
    co: float = Field(ge=0, le=100, description="CO (mg/m³)")
    ozone: float = Field(ge=0, le=1000, description="Ozone (µg/m³)")
    # Weather
    rh: float = Field(ge=0, le=100, description="Relative humidity (%)")
    ws: float = Field(ge=0, le=113, description="Wind speed (m/s)")
    sr: float = Field(ge=0, le=1500, description="Solar radiation (W/m²)")
    # Context
    state: str = Field(min_length=1)
    pm25_lag1: float = Field(ge=0, le=1000, description="Yesterday's PM2.5 (µg/m³)")
    aqi_lag1: float = Field(ge=0, le=1000, description="Yesterday's AQI")
    date: str = Field(description="YYYY-MM-DD")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


def get_season(month):
    if month in [12, 1, 2]: return 0
    if month in [3, 4, 5]: return 1
    if month in [6, 7, 8]: return 2
    return 3


# Human-readable labels for the raw model feature names, used in explanations.
FEATURE_LABELS = {
    "PM2.5 (ug/m3)": "PM2.5",
    "PM10 (ug/m3)": "PM10",
    "NO2 (ug/m3)": "NO₂",
    "SO2 (ug/m3)": "SO₂",
    "CO (mg/m3)": "CO",
    "Ozone (ug/m3)": "Ozone",
    "RH (%)": "Humidity",
    "WS (m/s)": "Wind Speed",
    "SR (W/mt2)": "Solar Radiation",
    "month": "Month",
    "day_of_year": "Day of Year",
    "season": "Season",
    "state_enc": "State",
    "PM2.5_lag1": "Yesterday's PM2.5",
    "AQI_lag1": "Yesterday's AQI",
}


def explain_prediction(row, top_n=5):
    """Return the top feature drivers for a single prediction.

    Uses XGBoost's built-in SHAP contributions (pred_contribs) — no extra
    dependency. Each driver's `impact` is how many AQI points that feature
    pushed the prediction up (positive) or down (negative) from the baseline.
    Returns [] if contributions can't be computed, so /predict never fails.
    """
    try:
        import xgboost as xgb

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(row, feature_names=list(row.columns))
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]

        # Last column is the baseline (bias) term — drop it.
        drivers = [
            {
                "feature": FEATURE_LABELS.get(col, col),
                "impact": round(float(contribs[i]), 1),
            }
            for i, col in enumerate(row.columns)
        ]
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return drivers[:top_n]
    except Exception:
        return []


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
        "category": get_aqi_category(prediction),
        "drivers": explain_prediction(row),
    }


def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"