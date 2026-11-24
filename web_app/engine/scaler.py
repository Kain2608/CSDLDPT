"""Z-score normalization engine using frozen training statistics (μ, σ).

IMPORTANT: Never refit on uploaded query audio. Always transform using the
exact mean (μ) and std (σ) computed over the 648 reference files in
`zscore_scaler_params.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from web_app.config import FEATURE_DIM, SCALER_PARAMS_JSON


@dataclass
class ScalerResult:
    """Structured output of Z-score normalization for UI reporting."""

    vector_raw: np.ndarray
    vector_zscore: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    scaling_time_ms: float


class ZScoreScaler:
    """Frozen Z-score scaler loaded from `zscore_scaler_params.json`."""

    def __init__(
        self,
        feature_names: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        params_path: Path | None = None,
    ) -> None:
        if len(feature_names) != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} feature names, got {len(feature_names)}.")

        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=np.float64)
        std_arr = np.asarray(std, dtype=np.float64)
        self.std = np.where(std_arr == 0.0, 1.0, std_arr)
        self.params_path = params_path

        if self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Scaler parameter shapes must be ({FEATURE_DIM},), got mean={self.mean.shape}, std={self.std.shape}."
            )

    @classmethod
    def from_json(cls, path: Path | str = SCALER_PARAMS_JSON) -> ZScoreScaler:
        """Load frozen μ and σ parameters from JSON."""
        params_path = Path(path)
        if not params_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp tham số Z-score: {params_path}")

        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        return cls(
            feature_names=payload["feature_names"],
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            params_path=params_path,
        )

    def transform_array(self, x: np.ndarray) -> np.ndarray:
        """Transform a 1D (92,) or 2D (N, 92) raw feature array using frozen μ and σ."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != FEATURE_DIM:
                raise ValueError(f"Expected vector of length {FEATURE_DIM}, got {arr.shape[0]}.")
            z = (arr - self.mean) / self.std
        elif arr.ndim == 2:
            if arr.shape[1] != FEATURE_DIM:
                raise ValueError(f"Expected matrix with {FEATURE_DIM} columns, got {arr.shape[1]}.")
            z = (arr - self.mean[None, :]) / self.std[None, :]
        else:
            raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, vector_raw: np.ndarray) -> ScalerResult:
        """Normalize a single 92D query vector and return detailed timing + stats."""
        raw = np.asarray(vector_raw, dtype=np.float64).reshape(-1)
        t0 = time.perf_counter()
        z = self.transform_array(raw)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ScalerResult(
            vector_raw=raw,
            vector_zscore=z,
            mean=self.mean.copy(),
            std=self.std.copy(),
            feature_names=list(self.feature_names),
            scaling_time_ms=elapsed_ms,
        )
