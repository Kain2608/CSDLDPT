"""Step 2 of the pipeline: copy & validate 92D dataset files from `output/` into `web_app/data/`."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.config import (
    DATA_DIR,
    FEATURE_DIM,
    METADATA_COLUMNS,
    RAW_FEATURES_CSV,
    SCALER_PARAMS_JSON,
    SOURCE_OUTPUT_DIR,
    ZSCORE_FEATURES_CSV,
)

REQUIRED_FILES = (
    "features_raw_92d.csv",
    "features_zscore_92d.csv",
    "zscore_scaler_params.json",
)


def prepare_data(force: bool = True) -> dict[str, int | float | str]:
    """Copy required CSV/JSON artifacts from `output/` to `web_app/data/` and validate integrity."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_FILES:
        src = SOURCE_OUTPUT_DIR / filename
        dst = DATA_DIR / filename
        if not src.is_file():
            raise FileNotFoundError(f"Thiếu tệp nguồn bắt buộc: {src}")
        if force or not dst.is_file():
            shutil.copy2(src, dst)

    # Validate copied files
    df_raw = pd.read_csv(RAW_FEATURES_CSV)
    df_zscore = pd.read_csv(ZSCORE_FEATURES_CSV)
    with open(SCALER_PARAMS_JSON, "r", encoding="utf-8") as f:
        scaler_params = json.load(f)

    feature_names = scaler_params["feature_names"]
    means = np.asarray(scaler_params["mean"], dtype=np.float64)
    stds = np.asarray(scaler_params["std"], dtype=np.float64)

    assert len(feature_names) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(feature_names)}"
    assert len(df_raw) == len(df_zscore), "Row count mismatch between raw and zscore CSVs"
    assert list(df_raw.columns[: len(METADATA_COLUMNS)]) == list(METADATA_COLUMNS), "Unexpected metadata columns"
    assert list(df_zscore.columns[len(METADATA_COLUMNS) :]) == feature_names, "Feature column mismatch"

    # Cross-check that (X_raw - mean) / std reproduces features_zscore_92d.csv accurately
    X_raw = df_raw[feature_names].to_numpy(dtype=np.float64)
    X_zscore = df_zscore[feature_names].to_numpy(dtype=np.float64)
    recomputed = (X_raw - means) / np.where(stds == 0, 1.0, stds)
    max_abs_err = float(np.max(np.abs(recomputed - X_zscore)))
    assert max_abs_err < 1e-9, f"Z-score verification failed: max_abs_err={max_abs_err}"

    summary = {
        "n_files": int(len(df_zscore)),
        "n_species": int(df_zscore["species"].nunique()),
        "n_features": int(len(feature_names)),
        "zscore_max_abs_error": max_abs_err,
        "data_dir": str(DATA_DIR),
    }
    print(f"[prepare_data] OK: {json.dumps(summary, ensure_ascii=False)}")
    return summary


if __name__ == "__main__":
    prepare_data()
