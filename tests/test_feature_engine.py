"""Tests for 92D audio feature extraction engine and input validation."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from web_app.config import (
    FEATURE_DIM,
    MIN_SAMPLES,
    RAW_FEATURES_CSV,
    SAMPLE_RATE,
    SCALER_PARAMS_JSON,
    resolve_audio_path,
)
from web_app.engine.feature_engine import (
    build_feature_columns,
    compute_92d_vector,
    extract_features_from_bytes,
    extract_features_from_path,
)


def _make_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_feature_columns_match_scaler_params() -> None:
    cols = build_feature_columns()
    assert len(cols) == FEATURE_DIM
    with open(SCALER_PARAMS_JSON, "r", encoding="utf-8") as f:
        params = json.load(f)
    assert cols == params["feature_names"]


def test_extract_from_stereo_44k_bytes_resamples_to_16k_mono() -> None:
    sr = 44100
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.2 * np.sin(2 * np.pi * 880.0 * t)
    stereo = np.stack([left, right], axis=1)

    wav_bytes = _make_wav_bytes(stereo, sr)
    res = extract_features_from_bytes(wav_bytes, "synthetic_test.wav")

    assert res.vector_raw.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(res.vector_raw))
    assert res.metadata.original_sample_rate == 44100
    assert res.metadata.resampled_sample_rate == SAMPLE_RATE
    assert res.metadata.original_channels == 2
    assert res.metadata.file_format == "WAV"
    assert abs(res.metadata.duration_sec - 0.5) < 0.02
    assert res.metadata.extraction_time_ms > 0.0


def test_reject_empty_and_too_short_audio() -> None:
    with pytest.raises(ValueError):
        compute_92d_vector(np.array([], dtype=np.float32), sr=SAMPLE_RATE)

    with pytest.raises(ValueError):
        compute_92d_vector(np.zeros(MIN_SAMPLES - 1, dtype=np.float32), sr=SAMPLE_RATE)


def test_minimum_valid_length_160_samples_does_not_crash() -> None:
    y = 0.2 * np.sin(np.linspace(0, 10, MIN_SAMPLES, dtype=np.float32))
    vec, n_frames = compute_92d_vector(y, sr=SAMPLE_RATE)
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(vec))
    assert n_frames >= 1


def test_parity_with_original_raw_features_csv() -> None:
    df_raw = pd.read_csv(RAW_FEATURES_CSV)
    row = df_raw.iloc[0]
    audio_path = resolve_audio_path(str(row["path"]))
    if audio_path is None:
        pytest.skip("Original WAV dataset directory not present on this host")

    res = extract_features_from_path(audio_path)
    expected = row[build_feature_columns()].to_numpy(dtype=np.float64)
    max_diff = float(np.max(np.abs(res.vector_raw - expected)))
    assert max_diff < 1e-3, f"Extracted 92D vector drifted from features_raw_92d.csv: max_diff={max_diff}"
