import json
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/cmapps/train_FD004.txt")

cols = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=cols)

UNIT_ID = 1
SEQ_LEN = 100

df_u = df[df["unit"] == UNIT_ID].sort_values("cycle").reset_index(drop=True)

if len(df_u) < SEQ_LEN:
    raise ValueError(f"unit={UNIT_ID} has only {len(df_u)} rows, need >= {SEQ_LEN}")

# Mid window first (avoid near-failure = low RUL)
START_CYCLE = 50
df_win = df_u[(df_u["cycle"] >= START_CYCLE) & (df_u["cycle"] < START_CYCLE + SEQ_LEN)].copy()

if len(df_win) < SEQ_LEN:
    df_win = df_u.tail(SEQ_LEN).copy()

batch_payload = {"records": df_win.to_dict(orient="records")}
Path("fd004_batch_swagger.json").write_text(json.dumps(batch_payload, indent=2))
print(" Wrote:", Path("fd004_batch_swagger.json").resolve())

single_payload = {
    "fd_name": "FD004",
    "unit": int(UNIT_ID),
    "rows": df_win.drop(columns=["unit"]).to_dict(orient="records"),
}
Path("fd004_single_swagger.json").write_text(json.dumps(single_payload, indent=2))
print(" Wrote:", Path("fd004_single_swagger.json").resolve())
