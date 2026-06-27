"""Tests for the AQI Prediction API.

Run from the repo root:

    pip install -r requirements.txt pytest httpx
    pytest -v
"""
import copy

import pytest
from fastapi.testclient import TestClient

from main import app, get_aqi_category, get_season

client = TestClient(app)

# A valid baseline payload; individual tests copy and tweak it.
VALID_PAYLOAD = {
    "pm25": 60, "pm10": 100, "no2": 20, "so2": 10, "co": 0.6, "ozone": 25,
    "rh": 50, "ws": 2, "sr": 200,
    "state": "Punjab",
    "pm25_lag1": 55, "aqi_lag1": 90,
    "date": "2026-06-27",
}


def payload(**overrides):
    p = copy.deepcopy(VALID_PAYLOAD)
    p.update(overrides)
    return p


# ---------- Simple endpoints ----------

def test_root_ok():
    res = client.get("/")
    assert res.status_code == 200
    assert "message" in res.json()


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ---------- /predict happy path ----------

def test_predict_returns_expected_shape():
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["predicted_aqi"], (int, float))
    assert body["category"] in {
        "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
    }
    # Explainability: a list of {feature, impact} dicts (possibly empty).
    assert isinstance(body["drivers"], list)
    for d in body["drivers"]:
        assert set(d.keys()) == {"feature", "impact"}
        assert isinstance(d["impact"], (int, float))


def test_unknown_state_is_accepted():
    # Unknown states encode to -1 rather than erroring.
    res = client.post("/predict", json=payload(state="Atlantis"))
    assert res.status_code == 200


# ---------- Validation rejections (422) ----------

@pytest.mark.parametrize("field,bad_value", [
    ("pm25", -5),        # below ge=0
    ("rh", 150),         # above le=100
    ("co", 500),         # above le=100
    ("ws", -1),          # below ge=0
    ("state", ""),       # empty string
])
def test_out_of_range_inputs_rejected(field, bad_value):
    res = client.post("/predict", json=payload(**{field: bad_value}))
    assert res.status_code == 422


@pytest.mark.parametrize("bad_date", ["27-06-2026", "2026/06/27", "not-a-date", ""])
def test_bad_date_format_rejected(bad_date):
    res = client.post("/predict", json=payload(date=bad_date))
    assert res.status_code == 422


def test_missing_field_rejected():
    p = payload()
    del p["pm25"]
    res = client.post("/predict", json=p)
    assert res.status_code == 422


# ---------- Pure helper functions ----------

@pytest.mark.parametrize("aqi,expected", [
    (0, "Good"), (50, "Good"),
    (51, "Satisfactory"), (100, "Satisfactory"),
    (101, "Moderate"), (200, "Moderate"),
    (201, "Poor"), (300, "Poor"),
    (301, "Very Poor"), (400, "Very Poor"),
    (401, "Severe"), (999, "Severe"),
])
def test_category_boundaries(aqi, expected):
    assert get_aqi_category(aqi) == expected


@pytest.mark.parametrize("month,season", [
    (1, 0), (2, 0), (12, 0),     # winter
    (3, 1), (5, 1),              # spring
    (6, 2), (8, 2),              # summer
    (9, 3), (11, 3),             # autumn
])
def test_season_mapping(month, season):
    assert get_season(month) == season
