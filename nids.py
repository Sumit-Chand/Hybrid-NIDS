"""Core library: features, signature engine, hybrid detector, SQLite alert store."""
import json, sqlite3, time
from contextlib import contextmanager
from pathlib import Path
import joblib, numpy as np, pandas as pd

# ---------------------------------------------------------------- features
FEATURES = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Flow Bytes/s", "Flow Packets/s", "Bwd Packets/s", "Packet Length Mean",
            "Packet Length Variance", "Fwd Header Length", "SYN Flag Count", "FIN Flag Count",
            "RST Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", "Init_Win_bytes_backward"]
SEVERITY = {"DoS": "high", "BruteForce": "high", "WebAttack": "high", "PortScan": "medium",
            "Bot": "critical", "BENIGN": "info"}

def map_label(l):
    """Collapse CIC-IDS2017 labels into 6 classes (None = dropped)."""
    l = str(l).strip().lower()
    if l == "benign": return "BENIGN"
    if "web attack" in l: return "WebAttack"
    if "patator" in l: return "BruteForce"
    if l == "bot": return "Bot"
    if "portscan" in l: return "PortScan"
    if "dos" in l or "ddos" in l or "heartbleed" in l: return "DoS"
    return None

def load_cic(folder, cap=40000):
    """Load + clean CIC-IDS2017 MachineLearningCVE CSVs (strip headers, drop inf/NaN, cap per class)."""
    frames = []
    for f in Path(folder).glob("*.csv"):
        d = pd.read_csv(f, low_memory=False, encoding="latin-1")
        d.columns = d.columns.str.strip()
        d["Label"] = d["Label"].map(map_label)
        frames.append(d[FEATURES + ["Label"]].dropna(subset=["Label"]))
    d = pd.concat(frames, ignore_index=True)
    d[FEATURES] = d[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d.dropna()
    return pd.concat([g.sample(min(len(g), cap), random_state=42) for _, g in d.groupby("Label")],
                     ignore_index=True)

# ------------------------------------------------- synthetic flow generator
def _flow(c, r):
    fwd = bwd = 1; port = 80; dur = 1e5; plen = 200.0; pf = 1.0; fin = 0; rst = 0
    iwf = iwb = 29200; ack = None
    if c == "BENIGN":
        port = int(r.choice([80, 443, 53, 8080, 3389, 25, 445, 993])); dur = r.lognormal(11, 2)
        fwd, bwd = r.poisson(8) + 1, r.poisson(8) + 1; plen = abs(r.normal(300, 150)) + 5
        pf = r.uniform(.5, 1.5); fin = int(r.random() < .7)
        iwf = int(r.choice([8192, 29200, 65535])); iwb = int(r.choice([229, 8192, 29200, 65535]))
    elif c == "DoS":
        port = int(r.choice([80, 443])); pf = r.uniform(.3, 1)
        if r.random() < .5:   # SYN-flood style: caught by signature
            dur = r.lognormal(6, 1); fwd = r.poisson(3) + 1; bwd = 0; plen = abs(r.normal(40, 20)); iwb = -1; ack = 0
        else:                 # slow DoS: only the ML model sees it
            dur = r.lognormal(15, 1); fwd = r.poisson(4) + 2; bwd = r.poisson(1); plen = abs(r.normal(120, 60))
    elif c == "PortScan":
        port = int(r.integers(1, 65535)); dur = r.lognormal(7, 1); fwd = 1; bwd = int(r.random() < .5)
        plen = abs(r.normal(0, 3)); rst = bwd; iwf = int(r.choice([1024, 2048, 4096])); iwb = 0 if bwd else -1; ack = 0
    elif c == "BruteForce":
        port = int(r.choice([21, 22])); dur = r.lognormal(13, .4); fwd = r.poisson(12) + 8; bwd = r.poisson(12) + 6
        plen = abs(r.normal(40, 12)); pf = r.uniform(.3, .8); fin = 1
    elif c == "WebAttack":
        port = 80; dur = r.lognormal(12, 1); fwd = r.poisson(10) + 5; bwd = r.poisson(10) + 5
        plen = abs(r.normal(450, 120)); pf = r.uniform(1.5, 3); iwb = 229; fin = 1
    elif c == "Bot":
        port = int(r.choice([8080, 443, 6667])); dur = r.lognormal(14, .3); fwd = r.poisson(3) + 2; bwd = r.poisson(3) + 2
        plen = abs(r.normal(100, 15)); pf = r.uniform(.1, .4); fin = 1
    n = fwd + bwd; sec = max(dur, 1) / 1e6
    src = f"10.0.{r.integers(0, 4)}.{r.integers(2, 250)}" if c == "BENIGN" else f"203.0.113.{r.integers(1, 40)}"
    return {"Source IP": src, "Destination IP": f"192.168.10.{r.choice([5, 10, 20, 50])}",
            "Destination Port": port, "Flow Duration": dur, "Total Fwd Packets": fwd, "Total Backward Packets": bwd,
            "Flow Bytes/s": n * plen / sec, "Flow Packets/s": n / sec, "Bwd Packets/s": bwd / sec,
            "Packet Length Mean": plen, "Packet Length Variance": (plen * pf) ** 2, "Fwd Header Length": fwd * 20,
            "SYN Flag Count": 1, "FIN Flag Count": fin, "RST Flag Count": rst,
            "ACK Flag Count": max(n - 1, 0) if ack is None else ack,
            "Init_Win_bytes_forward": iwf, "Init_Win_bytes_backward": iwb}

def gen_flows(n, mix=None, seed=None):
    r = np.random.default_rng(seed)
    mix = mix or {"BENIGN": .7, "DoS": .08, "PortScan": .08, "BruteForce": .05, "WebAttack": .05, "Bot": .04}
    labels = r.choice(list(mix), size=n, p=np.array(list(mix.values())) / sum(mix.values()))
    d = pd.DataFrame([_flow(l, r) for l in labels]); d["Label"] = labels
    return d

# --------------------------------------------------------- signature engine
# (rule id, attack class, vectorised condition over the flow features)
SIGNATURES = [
    ("SIG-1001 SYN flood", "DoS",
     lambda d: (d["Flow Packets/s"] > 5000) & (d["SYN Flag Count"] >= 1) & (d["Bwd Packets/s"] == 0)),
    ("SIG-2001 TCP SYN scan", "PortScan",
     lambda d: (d["Total Fwd Packets"] <= 2) & (d["SYN Flag Count"] >= 1) & (d["Packet Length Mean"] < 10)),
    ("SIG-3001 SSH/FTP brute force", "BruteForce",
     lambda d: d["Destination Port"].isin([21, 22]) & d["Total Fwd Packets"].between(15, 60)
               & (d["Flow Duration"] < 2e6) & (d["Packet Length Mean"] < 80)),
]

# ---------------------------------------------------------- hybrid detector
class Detector:
    def __init__(self, path="models/nids_model.joblib", threshold=0.6):
        b = joblib.load(path); self.pipe, self.enc, self.th = b["pipe"], b["enc"], threshold

    def inspect(self, df):
        X = (df[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
             .fillna(0).reset_index(drop=True))
        sig_cls = pd.Series([None] * len(X), dtype=object); sig_rule = sig_cls.copy()
        for name, cls, fn in SIGNATURES:                       # pass 1: deterministic rules
            m = fn(X) & sig_cls.isna(); sig_cls[m] = cls; sig_rule[m] = name
        P = self.pipe.predict_proba(X)                         # pass 2: ML classifier
        ml_cls, ml_conf = self.enc.inverse_transform(P.argmax(1)), P.max(1)
        out = []
        for s, rule, m, c in zip(sig_cls, sig_rule, ml_cls, ml_conf):   # consensus engine
            if isinstance(s, str) and s == m: out.append((s, "consensus", max(float(c), .95), rule))
            elif isinstance(s, str):          out.append((s, "signature", .90, rule))
            elif m != "BENIGN" and c >= self.th: out.append((m, "ml", float(c), ""))
            else:                             out.append(("BENIGN", "none", float(c), ""))
        res = pd.DataFrame(out, columns=["label", "engine", "confidence", "rule"])
        res["severity"] = res["label"].map(SEVERITY); res["is_alert"] = res["engine"] != "none"
        return res

# ------------------------------------------------------------- alert store
DB = "nids.db"

@contextmanager
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    try: yield c; c.commit()
    finally: c.close()

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
            src_ip TEXT, dst_ip TEXT, dst_port INTEGER, label TEXT, engine TEXT, rule TEXT,
            confidence REAL, severity TEXT, features TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS counters(k TEXT PRIMARY KEY, v INTEGER)")

def ingest(df, det):
    """Run both engines on a batch of flows, persist alerts, return the verdicts."""
    df = df.reset_index(drop=True); res = det.inspect(df); now = time.time()
    col = lambda k: df[k].astype(str).tolist() if k in df else ["-"] * len(df)
    src, dst = col("Source IP"), col("Destination IP")
    rows = []
    for i in res.index[res.is_alert]:
        r = res.loc[i]; feats = {k: float(df.at[i, k]) for k in FEATURES}
        rows.append((now, src[i], dst[i], int(df.at[i, "Destination Port"]), r.label, r.engine, r.rule,
                     round(r.confidence, 4), r.severity, json.dumps(feats)))
    with db() as c:
        c.executemany("INSERT INTO alerts(ts,src_ip,dst_ip,dst_port,label,engine,rule,confidence,severity,features)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
        c.execute("INSERT INTO counters VALUES('flows',?) ON CONFLICT(k) DO UPDATE SET v=v+excluded.v", (len(df),))
    return res

def stats():
    with db() as c:
        one = lambda q: (c.execute(q).fetchone() or [0])[0]
        by = lambda col: {r[0]: r[1] for r in c.execute(f"SELECT {col},COUNT(*) FROM alerts GROUP BY {col}")}
        tl = c.execute("SELECT strftime('%H:%M',ts,'unixepoch','localtime') m,COUNT(*) FROM alerts "
                       "GROUP BY m ORDER BY m DESC LIMIT 15").fetchall()
        top = c.execute("SELECT src_ip,COUNT(*) FROM alerts GROUP BY src_ip ORDER BY 2 DESC LIMIT 5").fetchall()
        return dict(total_flows=one("SELECT v FROM counters WHERE k='flows'"),
                    total_alerts=one("SELECT COUNT(*) FROM alerts"), by_class=by("label"), by_engine=by("engine"),
                    by_severity=by("severity"), timeline=[{"t": r[0], "n": r[1]} for r in tl][::-1],
                    top_sources=[{"ip": r[0], "n": r[1]} for r in top])

def recent(limit=50):
    with db() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,ts,src_ip,dst_ip,dst_port,label,engine,rule,confidence,severity FROM alerts "
            "ORDER BY id DESC LIMIT ?", (limit,))]

def clear():
    with db() as c:
        c.execute("DELETE FROM alerts"); c.execute("DELETE FROM counters")
