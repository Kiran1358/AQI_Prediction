# backend/app.py
from flask import Flask, request, jsonify, send_from_directory, abort
import os, json, joblib, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
MODEL_DIR = os.path.join(ROOT, "models")

REG = os.path.join(MODEL_DIR, "aqi_regressor.pkl")
SCALER = os.path.join(MODEL_DIR, "scaler.pkl")
FEAT = os.path.join(MODEL_DIR, "feature_cols.json")
METRICS = os.path.join(MODEL_DIR, "metrics.json")

if not (os.path.exists(REG) and os.path.exists(SCALER) and os.path.exists(FEAT)):
    print("Model artifacts missing. Run backend/clean_and_train.py first.")
    reg = None; scaler = None; feature_cols = []
    metrics = {}
else:
    reg = joblib.load(REG)
    scaler = joblib.load(SCALER)
    feature_cols = json.load(open(FEAT))
    metrics = json.load(open(METRICS)) if os.path.exists(METRICS) else {}

AQI_BUCKETS = [
    ("Good", 0, 50),
    ("Satisfactory", 51, 100),
    ("Moderate", 101, 200),
    ("Poor", 201, 300),
    ("Very Poor", 301, 400),
    ("Severe", 401, 500)
]
AQI_HEALTH = {
    "Good": "Minimal impact",
    "Satisfactory": "Minor breathing discomfort to sensitive people",
    "Moderate": "Breathing discomfort to people with lungs, asthma and heart diseases",
    "Poor": "Breathing discomfort to most people on prolonged exposure",
    "Very Poor": "Respiratory illness on prolonged exposure",
    "Severe": "Affects healthy people and seriously impacts those with existing diseases"
}

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")

@app.route("/")
def index():
    idx = os.path.join(FRONTEND, "index.html")
    if not os.path.exists(idx):
        return "<h3>index.html not found in frontend/</h3>", 404
    return send_from_directory(FRONTEND, "index.html")

@app.route("/css/<path:fn>")
def css(fn):
    p = os.path.join(FRONTEND, "css")
    if not os.path.exists(os.path.join(p, fn)):
        abort(404)
    return send_from_directory(p, fn)

@app.route("/js/<path:fn>")
def js(fn):
    p = os.path.join(FRONTEND, "js")
    if not os.path.exists(os.path.join(p, fn)):
        abort(404)
    return send_from_directory(p, fn)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if reg is None:
        return jsonify({"error": "Model not trained. Run clean_and_train.py first."}), 500

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "JSON body required"}), 400

    x = []
    for f in feature_cols:
        v = data.get(f, None)
        if v is None:
            v = data.get(f.replace(".", "_"), 0)
        try:
            x.append(float(v))
        except:
            x.append(0.0)

    X = np.array([x])
    Xs = scaler.transform(X)
    aqi_val = float(reg.predict(Xs)[0])

    bucket = "Unknown"
    for name, lo, hi in AQI_BUCKETS:
        if lo <= aqi_val <= hi:
            bucket = name
            break
    if bucket == "Unknown" and aqi_val > 500:
        bucket = "Severe"

    health = AQI_HEALTH.get(bucket, "")
    return jsonify({"aqi_value": round(aqi_val,3), "aqi_bucket": bucket, "health_message": health})

if __name__ == "__main__":
    print("Starting Flask server (frontend from):", FRONTEND)
    app.run(host="127.0.0.1", port=5000, debug=True)
