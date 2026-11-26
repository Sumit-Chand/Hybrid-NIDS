# Hybrid ML + Signature NIDS

    pip install -r requirements.txt
    python train.py            # trains on ./data/*.csv (CIC-IDS2017) or synthetic flows if none
    python app.py              # dashboard at http://127.0.0.1:5000

**Using the real dataset:** download the *MachineLearningCVE* CSVs from the UNB CIC-IDS2017 page,
put them in `data/`, and re-run `python train.py` (add `--model xgb` for XGBoost).
Metrics on the synthetic data are ~100% because it is easy by construction; only CIC-IDS2017 numbers are meaningful.
Re-check the thresholds in `nids.SIGNATURES` against real flows.

| File | Roadmap version |
|---|---|
| `train.py`, `nids.load_cic` | 1.0 preprocessing + ML baseline |
| `nids.SIGNATURES` | 2.0 rule engine |
| `nids.Detector` | 3.0 hybrid pipeline + consensus |
| `app.py`, `templates/index.html` | 4.0 Flask API + dashboard |
| SQLite (`nids.db`) in `nids.py` | 5.0 alert log / history |
| not included yet | 6.0 live capture (Scapy/CICFlowMeter -> `POST /api/analyze`) |

API: `GET /api/stats|alerts|metrics`, `POST /api/simulate|upload|analyze|clear`.
