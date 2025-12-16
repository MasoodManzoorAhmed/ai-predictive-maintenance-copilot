# backend/api/services/sequence_builder.py
"""
Sequence builder service.

Responsibilities:
- Build last-window sequences per engine (unit)
- Align with unified pipeline behavior:
  - For each unit, take the last `seq_len` cycles
  - If unit has < seq_len cycles, skip or pad (we SKIP by default)
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def build_last_window_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    skip_short: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - X_seq: (n_units, seq_len, n_features)
    - units: (n_units,) unit IDs

    Behavior:
    - Sort by unit, cycle
    - For each unit, slice last seq_len rows of feature_cols
    - If insufficient rows and skip_short=True => drop that unit
    """

    if unit_col not in df.columns or cycle_col not in df.columns:
        raise KeyError(f"Input df must contain '{unit_col}' and '{cycle_col}' columns.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    df_sorted = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)

    X_list = []
    units_list = []

    for unit_id, g in df_sorted.groupby(unit_col, sort=False):
        if len(g) < seq_len:
            if skip_short:
                continue
            # pad (rarely used; skip is safer)
            pad_len = seq_len - len(g)
            pad = np.repeat(g[feature_cols].iloc[[0]].values, pad_len, axis=0)
            seq = np.vstack([pad, g[feature_cols].values])
        else:
            seq = g[feature_cols].iloc[-seq_len:].values

        X_list.append(seq.astype(np.float32))
        units_list.append(int(unit_id))

    if not X_list:
        raise ValueError("No sequences built. Check seq_len or input data.")

    X_seq = np.stack(X_list, axis=0)
    units = np.array(units_list, dtype=np.int32)

    return X_seq, units
