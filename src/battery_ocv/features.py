"""Features available before OCV4; measurement time/units must be verified."""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS = ("OCV1", "OCV2", "OCV3")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use drops and their difference, not physical time-normalized derivatives."""
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OCV inputs: {sorted(missing)}")
    result = frame[list(INPUT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result["drop_12"] = result.OCV1 - result.OCV2
    result["drop_23"] = result.OCV2 - result.OCV3
    result["drop_change"] = result.drop_23 - result.drop_12
    result["std_123"] = result[list(INPUT_COLUMNS)].std(axis=1)
    return result


def encode_target(series: pd.Series) -> pd.Series:
    """Reject missing/unknown outcomes instead of silently assigning OK."""
    labels = series.astype("string").str.strip().str.upper()
    result = labels.map({"OK": 0, "NG": 1, "0": 0, "1": 1})
    if result.isna().any():
        raise ValueError(
            "Target contains missing or unknown labels; expected OK/NG or 0/1"
        )
    return result.astype(int)
