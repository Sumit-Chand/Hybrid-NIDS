"""Flask API + dashboard.   python app.py  ->  http://127.0.0.1:5000"""
import json
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
import nids

app = Flask(__name__)
nids.init_db()
if not Path("models/nids_model.joblib").exists():
    raise SystemExit("No model found. Run  python train.py  first.")
det = nids.Detector()

def summarize(df, res):
    out = dict(flows=len(df), alerts=int(res.is_alert.sum()), by_engine=res[res.is_alert].engine.value_counts().to_dict())
    if "Label" in df:   # ground truth available (simulation / labelled CSV)
        out["accuracy"] = round(float((df["Label"].reset_index(drop=True) == res.label).mean()), 4)
    return out

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/stats")
def api_stats(): return jsonify(nids.stats())

@app.get("/api/alerts")
def api_alerts(): return jsonify(nids.recent(min(int(request.args.get("limit", 50)), 500)))

@app.get("/api/metrics")
def api_metrics():
    p = Path("models/metrics.json")
    return jsonify(json.loads(p.read_text()) if p.exists() else {})

@app.post("/api/simulate")
def api_simulate():
    n = min(int((request.get_json(silent=True) or {}).get("n", 500)), 5000)
    df = nids.gen_flows(n); return jsonify(summarize(df, nids.ingest(df, det)))

@app.post("/api/analyze")
def api_analyze():
    """POST a JSON list of flow records (CIC-IDS2017 feature names)."""
    df = pd.DataFrame(request.get_json(force=True))
    miss = [c for c in nids.FEATURES if c not in df]
    if miss: return jsonify(error=f"Missing features: {miss}"), 400
    return jsonify(summarize(df, nids.ingest(df, det)))

@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f: return jsonify(error="Attach a CSV file as 'file'."), 400
    df = pd.read_csv(f); df.columns = df.columns.str.strip()
    miss = [c for c in nids.FEATURES if c not in df]
    if miss: return jsonify(error=f"CSV is missing columns: {miss}"), 400
    if "Label" in df: df["Label"] = df["Label"].map(nids.map_label).fillna("BENIGN")
    return jsonify(summarize(df, nids.ingest(df, det)))

@app.post("/api/clear")
def api_clear(): nids.clear(); return jsonify(ok=True)

if __name__ == "__main__":
    app.run(debug=True)
