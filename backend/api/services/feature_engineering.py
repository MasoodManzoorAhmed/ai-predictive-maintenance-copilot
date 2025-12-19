# backend/api/services/feature_engineering.py
"""
Feature engineering service.

Responsibilities:
- Apply rolling statistics and deltas per engine unit
- Keep logic consistent with the Colab unified pipeline

- Rolling features must be computed per unit (engine).
- We build new columns in a dict then concat once (avoids fragmentation).
"""

from typing import List

import numpy as np
import pandas as pd


def apply_rolling_and_delta_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    win_short: int = 3,
    win_long: int = 5,
) -> pd.DataFrame:
    """
    Adds:
    - sensor_roll{win_short}_mean/std
    - sensor_roll{win_long}_mean/std
    - sensor_delta (per-cycle difference)
    Computed per engine unit.

    Returns a NEW dataframe (does not mutate input).
    """

    if unit_col not in df.columns or cycle_col not in df.columns:
        raise KeyError(f"Input df must contain '{unit_col}' and '{cycle_col}' columns.")

    # Ensure stable ordering for rolling operations
    df_sorted = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)

    # Group once (fast)
    grp = df_sorted.groupby(unit_col, sort=False)

    new_cols = {}

    for s in sensor_cols:
        if s not in df_sorted.columns:
            # If a sensor is missing, that's a hard error: config and input mismatch
            raise KeyError(f"Missing required sensor column: {s}")

        # rolling mean/std (short)
        roll_short = grp[s].rolling(window=win_short, min_periods=1)
        new_cols[f"{s}_roll{win_short}_mean"] = roll_short.mean().reset_index(level=0, drop=True).astype(np.float32)
        new_cols[f"{s}_roll{win_short}_std"] = roll_short.std(ddof=0).reset_index(level=0, drop=True).fillna(0).astype(np.float32)

        # rolling mean/std (long)
        roll_long = grp[s].rolling(window=win_long, min_periods=1)
        new_cols[f"{s}_roll{win_long}_mean"] = roll_long.mean().reset_index(level=0, drop=True).astype(np.float32)
        new_cols[f"{s}_roll{win_long}_std"] = roll_long.std(ddof=0).reset_index(level=0, drop=True).fillna(0).astype(np.float32)

        # delta per unit
        new_cols[f"{s}_delta"] = grp[s].diff().fillna(0).astype(np.float32)

    fe_df = pd.concat([df_sorted, pd.DataFrame(new_cols)], axis=1)

    return fe_df
