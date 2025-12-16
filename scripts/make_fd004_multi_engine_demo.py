# make_fd004_multi_engine_demo.py
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

# =========================
# CONFIG
# =========================
FD_NAME = "FD004"
RAW_PATH = Path("data/cmapps") / f"train_{FD_NAME}.txt"   # preferred path
OUT_DIR = Path("data")

N_UNITS = 10                  # how many engines to include
SEQ_LEN = 100                 # FD004 model uses 100
SEED = 42

# CMAPSS column names (train file)
COLS = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

# =========================
# LOAD
# =========================
def find_raw_file() -> Path:
    if RAW_PATH.exists():
        return RAW_PATH
    matches = list(Path(".").rglob(f"train_{FD_NAME}.txt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find train_{FD_NAME}.txt\n"
            f"Expected: {RAW_PATH}\n"
            f"Try placing it under: data/cmapps/"
        )
    return matches[0]

raw = find_raw_file()
df = pd.read_csv(raw, sep=r"\s+", header=None, names=COLS)

# Ensure sorted
df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
df = df.dropna(subset=["cycle"]).sort_values(["unit", "cycle"]).reset_index(drop=True)

# =========================
# PICK UNITS THAT HAVE >= SEQ_LEN ROWS
# =========================
counts = df.groupby("unit").size()
eligible_units = counts[counts >= SEQ_LEN].index.tolist()

if len(eligible_units) < N_UNITS:
    raise ValueError(
        f"Not enough eligible units with >= {SEQ_LEN} cycles.\n"
        f"Eligible: {len(eligible_units)}"
    )

random.seed(SEED)
picked_units = sorted(random.sample(eligible_units, N_UNITS))

# =========================
# BUILD MULTI-ENGINE DATA (last SEQ_LEN cycles per unit)
# =========================
parts = []
for u in picked_units:
    d_u = df[df["unit"] == u].copy()
    last_window = d_u.tail(SEQ_LEN).copy()
    parts.append(last_window)

out_df = pd.concat(parts, ignore_index=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) CSV output (useful for Streamlit uploads)
out_csv = OUT_DIR / f"multi_{FD_NAME}_{N_UNITS}units_tail{SEQ_LEN}.csv"
out_df.to_csv(out_csv, index=False)

# 2) JSON output for FastAPI batch endpoint: POST /predict/fd004
payload = {"records": out_df.to_dict(orient="records")}
out_json = OUT_DIR / f"fd004_multi_engine_{N_UNITS}units_tail{SEQ_LEN}.json"
out_json.write_text(json.dumps(payload, indent=2))

print("✅ Multi-engine demo CSV created:")
print("   ", out_csv.resolve())
print("✅ Multi-engine demo JSON created (for API):")
print("   ", out_json.resolve())
print(f"✅ Units included: {picked_units}")
print(f"✅ Rows: {len(out_df)} (should be {N_UNITS * SEQ_LEN})")
