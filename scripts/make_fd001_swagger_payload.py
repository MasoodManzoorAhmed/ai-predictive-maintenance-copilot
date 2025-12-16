import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/cmapps/train_FD001.txt")

# CMAPSS columns (no header in file)
cols = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

# Read whitespace-separated file
df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=cols)

# Pick a real unit (change this if you want)
UNIT_ID = 1

# FD001 seq_len is typically 30
SEQ_LEN = 30

df_u = df[df["unit"] == UNIT_ID].sort_values("cycle").reset_index(drop=True)

if len(df_u) < SEQ_LEN:
    raise ValueError(f"unit={UNIT_ID} has only {len(df_u)} rows, need >= {SEQ_LEN}")

# Use last 30 cycles (most realistic)
START_CYCLE = 50
df_win = df_u[(df_u["cycle"] >= START_CYCLE) & (df_u["cycle"] < START_CYCLE + SEQ_LEN)].copy()

if len(df_win) < SEQ_LEN:
    raise ValueError(f"Not enough rows for unit={UNIT_ID} starting at cycle={START_CYCLE}")

# -------- Batch endpoint payload (/predict/fd001) --------
batch_payload = {"records": df_win.to_dict(orient="records")}

out_batch = Path("fd001_batch_swagger.json")
out_batch.write_text(json.dumps(batch_payload, indent=2))
print("✅ Wrote:", out_batch.resolve())

# -------- Single-engine payload (/single/predict) --------
single_payload = {
    "fd_name": "FD001",
    "unit": int(UNIT_ID),
    "rows": df_win.drop(columns=["unit"]).to_dict(orient="records"),
}

out_single = Path("fd001_single_swagger.json")
out_single.write_text(json.dumps(single_payload, indent=2))
print("✅ Wrote:", out_single.resolve())

print("\nTip: Open the JSON file and paste directly into Swagger.")
