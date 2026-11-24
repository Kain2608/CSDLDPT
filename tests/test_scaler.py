"""Tests for frozen Z-score scaler using zscore_scaler_params.json."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from web_app.config import FEATURE_DIM, RAW_FEATURES_CSV, SCALER_PARAMS_JSON, ZSCORE_FEATURES_CSV
from web_app.engine.scaler import ZScoreScaler


def test_scaler_reproduces_zscore_csv_exactly() -> None:
    scaler = ZScoreScaler.from_json(SCALER_PARAMS_JSON)
    df_raw = pd.read_csv(RAW_FEATURES_CSV)
    df_z = pd.read_csv(ZSCORE_FEATURES_CSV)

    cols = scaler.feature_names
    X_raw = df_raw[cols].to_numpy(dtype=np.float64)
    X_z_expected = df_z[cols].to_numpy(dtype=np.float64)

    X_z_computed = scaler.transform_array(X_raw)
    max_err = float(np.max(np.abs(X_z_computed - X_z_expected)))
    assert max_err < 1e-9, f"Z-score mismatch: max_err={max_err}"


def test_scaler_single_vector_transform_and_validation() -> None:
    scaler = ZScoreScaler.from_json(SCALER_PARAMS_JSON)
    res = scaler.transform(scaler.mean)
    assert res.vector_zscore.shape == (FEATURE_DIM,)
    assert np.allclose(res.vector_zscore, 0.0, atol=1e-12)
    assert res.scaling_time_ms >= 0.0

    with pytest.raises(ValueError):
        scaler.transform(np.zeros(50))
