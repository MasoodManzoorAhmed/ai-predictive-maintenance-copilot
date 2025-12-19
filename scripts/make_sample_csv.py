# make_sample_csv.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================
# CONFIG
# =============================
FDS = ["FD001", "FD002", "FD003", "FD004"]

# Where raw CMAPSS train files are 
# data/cmapps/train_FD001.txt etc.
RAW_DIR_PREFERRED = Path("data") / "cmapps"

# Output folder for generated recruiter demo CSVs
OUT_DIR = Path("data")

# Demo windows
WINDOW_50 = 50
WINDOW_100 = 100  # REQUIRED for batch endpoints for FD002/FD003/FD004

# If you REALLY want unit=1 only, set AUTO_PICK_UNIT=False and FORCE_UNIT_ID=1
AUTO_PICK_UNIT = True
FORCE_UNIT_ID = 1

# CMAPSS column names
COLS = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

# =============================
# HELPERS
# =============================
def choose_best_match(matches: list[Path]) -> Path:
    
    def score(p: Path) -> tuple:
        s = str(p).lower()
        has_cmapss = ("cmapss" in s) or ("cmapps" in s)
        return (0 if has_cmapss else 1, len(s))

    return sorted(matches, key=score)[0]


def find_raw_file(fd_name: str) -> Path:
    preferred = RAW_DIR_PREFERRED / f"train_{fd_name}.txt"
    if preferred.exists():
        return preferred

    target = f"train_{fd_name}.txt"
    matches = list(Path(".").rglob(target))
    if not matches:
        raise FileNotFoundError(
            f"❌ Could not find {target}\n"
            f"Expected (preferred): {preferred}\n"
            f"Try Windows: dir /s {target}"
        )

    if len(matches) > 1:
        print(f"⚠️ Multiple matches found for {target}:")
        for m in matches:
            print("   -", m.resolve())
        chosen = choose_best_match(matches)
        print("✅ Using:", chosen.resolve())
        return chosen

    return matches[0]


def load_fd_dataframe(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=COLS)
    if df.shape[1] != len(COLS):
        raise ValueError(
            f"❌ Column mismatch reading {raw_path}\n"
            f"Got shape={df.shape}, expected {len(COLS)} columns.\n"
            f"Check file format / wrong file."
        )
    return df


def pick_unit_with_min_cycles(df: pd.DataFrame, min_cycles: int) -> int:
    """
    Picks the FIRST unit that has at least min_cycles rows.
    If none exist, pick the longest unit (still returns something).
    """
    counts = df.groupby("unit").size().sort_index()
    valid = counts[counts >= min_cycles]
    if not valid.empty:
        return int(valid.index[0])
    return int(counts.idxmax())


def save_window_csv(df_unit: pd.DataFrame, fd_name: str, unit_id: int, mode: str, window: int) -> Path:
    part = df_unit.head(window) if mode == "head" else df_unit.tail(window)
    out = OUT_DIR / f"sample_{fd_name}_unit{unit_id}_{mode}{window}.csv"
    part.to_csv(out, index=False)
    return out


def generate_samples_for_fd(fd_name: str):
    print("\n==============================")
    print(f"📦 Processing {fd_name}")
    print("==============================")

    raw_path = find_raw_file(fd_name)
    print("✅ Found:", raw_path.resolve())

    df = load_fd_dataframe(raw_path)
    total_units = df["unit"].nunique()
    print(f"Rows total: {len(df):,} | Units: {total_units}")

    # FD rules: batch windows
    requires_100 = fd_name in ["FD002", "FD003", "FD004"]
    min_needed = WINDOW_100 if requires_100 else WINDOW_50

    unit_id = pick_unit_with_min_cycles(df, min_needed) if AUTO_PICK_UNIT else FORCE_UNIT_ID

    df_unit = (
        df[df["unit"] == unit_id]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    print(
        f"Using unit={unit_id} | rows={len(df_unit)} | "
        f"cycle range: {int(df_unit['cycle'].min())}→{int(df_unit['cycle'].max())}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Always save head50/tail50 (nice for UI demo)
    if len(df_unit) >= WINDOW_50:
        head50 = save_window_csv(df_unit, fd_name, unit_id, "head", WINDOW_50)
        tail50 = save_window_csv(df_unit, fd_name, unit_id, "tail", WINDOW_50)
        print(f"🟢 head50 saved: {head50}")
        print(f"🔴 tail50 saved: {tail50}")
    else:
        print(f"⚠️ Cannot create head50/tail50: only {len(df_unit)} rows.")
        return

    # For FD002/FD003/FD004 also save head100/tail100 (needed for batch endpoint)
    if requires_100:
        if len(df_unit) >= WINDOW_100:
            head100 = save_window_csv(df_unit, fd_name, unit_id, "head", WINDOW_100)
            tail100 = save_window_csv(df_unit, fd_name, unit_id, "tail", WINDOW_100)
            print(f"🟢 head100 saved: {head100}")
            print(f"🔴 tail100 saved: {tail100}")
        else:
            print(f"⚠️ Cannot create head100/tail100: only {len(df_unit)} rows. (Batch will fail.)")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    print("✅ Generating CMAPSS sample CSVs...")
    print("AUTO_PICK_UNIT =", AUTO_PICK_UNIT)
    print("FORCE_UNIT_ID  =", FORCE_UNIT_ID)

    for fd in FDS:
        generate_samples_for_fd(fd)

    print("\n✅ Done. Check the /data folder for sample_*.csv files.")
