import { useState } from "react";
import axios from "axios";
import "./App.css";

// Configurable via VITE_API_URL at build time; falls back to the deployed API.
const API_URL =
  import.meta.env.VITE_API_URL ||
  "https://aqi-prediction-l2gz.onrender.com/predict";

const STATES = [
  "Punjab", "Delhi", "Maharashtra", "Uttar Pradesh", "Karnataka",
  "Tamil Nadu", "West Bengal", "Gujarat", "Rajasthan", "Haryana"
];

const CATEGORY_COLOR = {
  Good: "#4ade80",
  Satisfactory: "#a3e635",
  Moderate: "#fbbf24",
  Poor: "#fb923c",
  "Very Poor": "#f87171",
  Severe: "#dc2626",
};

// Circular gauge: maps an AQI value (0–500) onto a 270° arc.
function Gauge({ value, category }) {
  const color = CATEGORY_COLOR[category] || "#94a3b8";
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const arcFraction = 0.75; // 270° of a full circle
  const pct = Math.max(0, Math.min(value / 500, 1));
  const dash = circumference * arcFraction;
  const filled = dash * pct;

  return (
    <div className="gauge">
      <svg viewBox="0 0 200 200" className="gauge-svg">
        {/* track */}
        <circle
          cx="100" cy="100" r={radius}
          fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth="14"
          strokeLinecap="round" strokeDasharray={`${dash} ${circumference}`}
          transform="rotate(135 100 100)"
        />
        {/* value arc */}
        <circle
          cx="100" cy="100" r={radius}
          fill="none" stroke={color} strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          transform="rotate(135 100 100)"
          style={{ transition: "stroke-dasharray 0.8s ease, stroke 0.4s ease" }}
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-value" style={{ color }}>{value}</span>
        <span className="gauge-label">AQI</span>
      </div>
    </div>
  );
}

// Diverging bar chart of the top features driving a prediction.
// Positive impact (pushes AQI up) = red-ish; negative (pulls down) = green.
function Drivers({ drivers }) {
  if (!drivers || drivers.length === 0) return null;
  const max = Math.max(...drivers.map((d) => Math.abs(d.impact)), 1);

  return (
    <div className="drivers">
      <h3 className="drivers-title">What's driving this AQI</h3>
      <ul className="drivers-list">
        {drivers.map((d) => {
          const up = d.impact >= 0;
          const width = (Math.abs(d.impact) / max) * 100;
          return (
            <li key={d.feature} className="driver-row">
              <span className="driver-name">{d.feature}</span>
              <span className="driver-track">
                <span
                  className={"driver-bar " + (up ? "up" : "down")}
                  style={{ width: `${width}%` }}
                />
              </span>
              <span className={"driver-val " + (up ? "up" : "down")}>
                {up ? "+" : "−"}{Math.abs(d.impact)}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="drivers-foot">
        Points each factor pushed the AQI up (+) or down (−) from the baseline.
      </p>
    </div>
  );
}

function App() {
  const [form, setForm] = useState({
    pm25: 60,
    pm10: 100,
    no2: 20,
    so2: 10,
    co: 0.6,
    ozone: 25,
    rh: 50,
    ws: 2,
    sr: 200,
    state: "Punjab",
    pm25_lag1: 55,
    aqi_lag1: 90,
    date: new Date().toISOString().split("T")[0],
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === "state" || name === "date" ? value : parseFloat(value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await axios.post(API_URL, form);
      setResult(res.data);
    } catch {
      setError("Could not get prediction. The server may be waking up — try again in 30 seconds.");
    } finally {
      setLoading(false);
    }
  };

  const accent = result ? CATEGORY_COLOR[result.category] : "#38bdf8";

  return (
    <div
      className="page"
      style={{ "--accent": accent }}
      data-category={result?.category || "none"}
    >
      <div className="glow" />
      <main className="card">
        <header className="card-head">
          <h1>AQI Predictor</h1>
          <p className="subtitle">Predict India's Air Quality Index using ML</p>
        </header>

        <form onSubmit={handleSubmit} className="form">
          <fieldset className="section">
            <legend>Pollutants</legend>
            <div className="grid">
              <label>
                PM2.5 <span className="unit">µg/m³</span>
                <input type="number" name="pm25" value={form.pm25} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
              <label>
                PM10 <span className="unit">µg/m³</span>
                <input type="number" name="pm10" value={form.pm10} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
              <label>
                NO₂ <span className="unit">µg/m³</span>
                <input type="number" name="no2" value={form.no2} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
              <label>
                SO₂ <span className="unit">µg/m³</span>
                <input type="number" name="so2" value={form.so2} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
              <label>
                CO <span className="unit">mg/m³</span>
                <input type="number" name="co" value={form.co} onChange={handleChange} step="0.1" min="0" max="100" required />
              </label>
              <label>
                Ozone <span className="unit">µg/m³</span>
                <input type="number" name="ozone" value={form.ozone} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
            </div>
          </fieldset>

          <fieldset className="section">
            <legend>Weather</legend>
            <div className="grid">
              <label>
                Humidity <span className="unit">%</span>
                <input type="number" name="rh" value={form.rh} onChange={handleChange} step="0.1" min="0" max="100" required />
              </label>
              <label>
                Wind Speed <span className="unit">m/s</span>
                <input type="number" name="ws" value={form.ws} onChange={handleChange} step="0.1" min="0" max="113" required />
              </label>
              <label>
                Solar Radiation <span className="unit">W/m²</span>
                <input type="number" name="sr" value={form.sr} onChange={handleChange} step="0.1" min="0" max="1500" required />
              </label>
            </div>
          </fieldset>

          <fieldset className="section">
            <legend>Location &amp; Context</legend>
            <div className="grid">
              <label>
                State
                <select name="state" value={form.state} onChange={handleChange}>
                  {STATES.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </label>
              <label>
                Date
                <input type="date" name="date" value={form.date} onChange={handleChange} required />
              </label>
              <label>
                Yesterday's PM2.5 <span className="unit">µg/m³</span>
                <input type="number" name="pm25_lag1" value={form.pm25_lag1} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
              <label>
                Yesterday's AQI
                <input type="number" name="aqi_lag1" value={form.aqi_lag1} onChange={handleChange} step="0.1" min="0" max="1000" required />
              </label>
            </div>
          </fieldset>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting…" : "Predict AQI"}
          </button>

          {loading && (
            <p className="hint">
              Waking the model server… the first request after idle can take up to
              ~30s on the free tier.
            </p>
          )}
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <Gauge value={result.predicted_aqi} category={result.category} />
            <div
              className="category-pill"
              style={{ background: CATEGORY_COLOR[result.category] }}
            >
              {result.category}
            </div>
            <div className="scale">
              {Object.entries(CATEGORY_COLOR).map(([name, color]) => (
                <div
                  key={name}
                  className={"scale-seg" + (name === result.category ? " active" : "")}
                  style={{ background: color }}
                  title={name}
                />
              ))}
            </div>

            <Drivers drivers={result.drivers} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
