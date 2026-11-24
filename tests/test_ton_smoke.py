"""Smoke test of the TON_IoT pipeline on synthetic data: preparation, one federated continual run with all memory
actions (BRMA, BRMA-M, BRMA-G, BRMA-C, controls), and the summary metrics. Runs in about a minute on a laptop."""
import os, sys, shutil, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; TMP = Path(__file__).resolve().parent / "_tmp"
shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir()
os.environ["TON_ROUNDS"] = "1"

# synthetic flows with the TON_IoT column layout
g = np.random.default_rng(0); parts = []
for ci, (c, n) in enumerate({"normal": 3000, "scanning": 300, "dos": 300, "ddos": 800, "password": 300, "backdoor": 300,
                             "injection": 300, "xss": 300, "ransomware": 300, "mitm": 60}.items()):
    parts.append(pd.DataFrame({"ts": g.integers(1e9, 2e9, n), "src_ip": "10.0.0.2", "src_port": g.integers(1, 65535, n), "dst_ip": "10.0.0.1",
                               "dst_port": g.choice([80, 53, 443, 21 + ci], n), "proto": g.choice(["tcp", "udp"], n), "duration": g.exponential(1 + ci, n),
                               "src_bytes": g.integers(0, 10 ** (2 + ci % 4), n), "dst_bytes": g.integers(0, 10 ** (1 + ci % 3), n),
                               "conn_state": g.choice(["S0", "SF", "REJ"], n), "src_pkts": g.poisson(3 + ci, n), "label": int(ci > 0), "type": c}))
csv = TMP / "train_test_network.csv"; pd.concat(parts).sample(frac=1, random_state=0).to_csv(csv, index=False)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
T = load("ton_study", ROOT / "src/ton/ton_study.py"); I = load("ton_improve", ROOT / "src/ton/ton_improve.py")
I.TON_CFG = dict(rounds=1, capacity=500, features="ports")
rep = T.prepare(csv, TMP / "data", ports=True); print("prepared:", rep["rows"], "rows,", rep["features"], "features")
for arm in ("C", "K3H"):
    keys = I.worker(dict(seed=101, arm=arm, data=str(TMP / "data"), out=str(TMP / "out")))
    print(arm, "->", len(keys), "runs")
df = I.table(TMP / "out", "test")
assert len(df) == 2 * len(I.BRANCH)
assert df[df.method.isin(["BRMA", "BRMA-M", "BRMA-G", "BRMA-C"])].false_removal.max() <= 0.12   # conformal audits only
print(df[df.arm == "K3H"][["method", "H", "detection", "false_removal", "macro_f1"]].round(3).to_string(index=False))
shutil.rmtree(TMP, ignore_errors=True)
print("SMOKE TEST PASSED")
