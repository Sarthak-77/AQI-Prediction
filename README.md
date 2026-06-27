# AQI Predictor 🌫️

A full-stack machine-learning app that predicts India's **Air Quality Index (AQI)**
from pollutant concentrations, weather conditions, and location, then classifies
the result on the official CPCB scale (Good → Severe).

> **Live demo:** _add your deployed frontend URL here_
> **API:** https://aqi-prediction-l2gz.onrender.com

<!-- Add a screenshot or GIF of the app here — it's the single biggest thing reviewers look at -->
<!-- ![AQI Predictor screenshot](aqi-frontend/src/assets/hero.png) -->

---

## What it does

You enter 13 readings (pollutants like PM2.5/PM10/NO₂, weather like humidity and
wind speed, plus state, date, and yesterday's values). A trained gradient-boosted
model returns a predicted AQI value and its category, color-coded for severity.

## Tech stack

| Layer      | Tools |
|------------|-------|
| Frontend   | React 19, Vite 8, Axios |
| Backend    | FastAPI, Pydantic, Uvicorn |
| ML         | XGBoost, scikit-learn, pandas, NumPy, joblib |
| Hosting    | Render (API) |

## How it works

1. The React form collects 13 inputs and `POST`s them to `/predict`.
2. FastAPI validates the payload with a Pydantic schema.
3. The backend derives extra features from the date — `month`, `day_of_year`,
   `season` — and encodes the state with a saved mapping.
4. The pre-trained XGBoost model (`model/aqi_model_india.pkl`) predicts the AQI.
5. The value is mapped to a CPCB category and returned to the UI.

### AQI categories

| Range    | Category      |
|----------|---------------|
| 0–50     | Good          |
| 51–100   | Satisfactory  |
| 101–200  | Moderate      |
| 201–300  | Poor          |
| 301–400  | Very Poor     |
| 401+     | Severe        |

## Running locally

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# API at http://127.0.0.1:8000  ·  docs at http://127.0.0.1:8000/docs
```

### Frontend
```bash
cd aqi-frontend
npm install
npm run dev
# App at http://localhost:5173
```

> To run fully local, set the API URL via env var:
> ```bash
> cd aqi-frontend
> cp .env.example .env      # then edit VITE_API_URL → http://127.0.0.1:8000/predict
> ```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Covers the happy path, the explainability payload shape, validation rejections
(out-of-range values, bad dates, missing fields), and the AQI category / season
boundary logic.

## Deployment

- **API** is hosted on Render.
- **Frontend** is configured for Vercel (`aqi-frontend/vercel.json`). Deploy with:
  ```bash
  cd aqi-frontend
  npx vercel            # set VITE_API_URL in the Vercel project env vars
  ```

## API reference

### `POST /predict`

Request body:
```json
{
  "pm25": 60, "pm10": 100, "no2": 20, "so2": 10, "co": 0.6, "ozone": 25,
  "rh": 50, "ws": 2, "sr": 200,
  "state": "Punjab",
  "pm25_lag1": 55, "aqi_lag1": 90,
  "date": "2026-06-27"
}
```

Response:
```json
{
  "predicted_aqi": 142.3,
  "category": "Moderate",
  "drivers": [
    { "feature": "PM2.5", "impact": 38.4 },
    { "feature": "Yesterday's AQI", "impact": 21.7 },
    { "feature": "PM10", "impact": 12.1 },
    { "feature": "Humidity", "impact": -6.5 },
    { "feature": "Wind Speed", "impact": -9.2 }
  ]
}
```

`drivers` are the top features behind each prediction, computed with XGBoost's
built-in SHAP contributions. `impact` is how many AQI points that feature pushed
the result up (+) or down (−) from the model baseline — surfaced in the UI as a
"what's driving this AQI" chart.

Other endpoints: `GET /` (status) and `GET /health` (health check).

## Project structure
```
.
├── main.py              FastAPI app + model inference
├── model/               Pickled XGBoost model, feature list, state encoder
├── requirements.txt     Python dependencies
└── aqi-frontend/        React + Vite single-page app
    └── src/App.jsx       The prediction form and result UI
```

## Roadmap

- [x] Modern glassmorphism UI with an AQI gauge and category scale
- [x] Feature explainability (which inputs drove the prediction)
- [x] Input range validation on both client and server
- [x] Unit tests for `/predict`
- [ ] Deploy the frontend and add the live link above
