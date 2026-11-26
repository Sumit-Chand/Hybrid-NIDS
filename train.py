"""Train the ML classifier. Uses CIC-IDS2017 CSVs in ./data if present, else synthetic flows.
   python train.py [--data data] [--model rf|xgb] [--n 30000]"""
import argparse, json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import nids

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
ap.add_argument("--n", type=int, default=30000, help="synthetic flows when no CSVs are found")
a = ap.parse_args()

if Path(a.data).exists() and list(Path(a.data).glob("*.csv")):
    df, source = nids.load_cic(a.data), "CIC-IDS2017"
else:
    df, source = nids.gen_flows(a.n, seed=1), "synthetic (no CSVs in ./data)"
print(f"Dataset: {source} | {len(df):,} flows\n{df['Label'].value_counts().to_string()}\n")

enc = LabelEncoder(); y = enc.fit_transform(df["Label"])
Xtr, Xte, ytr, yte = train_test_split(df[nids.FEATURES], y, test_size=.25, stratify=y, random_state=42)
if a.model == "xgb":
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=200, max_depth=8, n_jobs=-1, eval_metric="mlogloss")
else:
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced", random_state=42)
pipe = make_pipeline(MinMaxScaler(), clf).fit(Xtr, ytr)
pred = pipe.predict(Xte)

p, r, f, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
metrics = dict(source=source, model=a.model, accuracy=accuracy_score(yte, pred), precision=p, recall=r, f1=f,
               labels=list(enc.classes_), confusion=confusion_matrix(yte, pred).tolist())
print(classification_report(yte, pred, target_names=enc.classes_, zero_division=0))
Path("models").mkdir(exist_ok=True)
joblib.dump({"pipe": pipe, "enc": enc}, "models/nids_model.joblib")
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print("Saved models/nids_model.joblib and models/metrics.json")
