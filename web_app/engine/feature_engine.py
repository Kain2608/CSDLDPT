"""Audio feature extraction engine producing the 92D Feature Set C vector.

Strictly mirrors the DSP formulation in `extract_features.py`:
- Mono resampling to 16,000 Hz
- STFT with Hann window, n_fft=512, win_length=400 (25 ms), hop_length=160 (10 ms)
- Set A (14D): rms, zcr, centroid, bandwidth, rolloff, flatness, flux x (mean, std)
- MFCC (26D): 13 coefficients x (mean, std)
- Delta MFCC (26D): 13 coefficients x (mean_abs, std)
- Delta-Delta MFCC (26D): 13 coefficients x (mean_abs, std)
Total = 92 dimensions.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from web_app.config import (
    DELTA_WIDTH,
    FEATURE_DIM,
    HOP_LENGTH,
    MIN_SAMPLES,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
    WIN_LENGTH,
)


@dataclass
class AudioMetadata:
    """Technical metadata of an input audio file."""

    filename: str
    file_format: str
    file_size_bytes: int
    original_sample_rate: int
    resampled_sample_rate: int
    original_channels: int
    duration_sec: float
    num_samples: int
    num_frames_hop: int
    num_stft_frames: int
    extraction_time_ms: float


@dataclass
class FeatureExtractionResult:
    """Output of 92D audio feature extraction."""

    vector_raw: np.ndarray
    feature_names: list[str]
    metadata: AudioMetadata
    waveform: np.ndarray


def build_feature_columns() -> list[str]:
    """Generate the canonical 92 feature column names matching `extract_features.py`."""
    basic = ["rms", "zcr", "centroid", "bandwidth", "rolloff", "flatness", "flux"]
    cols = [f"{name}_{stat}" for name in basic for stat in ("mean", "std")]
    cols += [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    cols += [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    cols += [f"delta_{i}_mean_abs" for i in range(1, N_MFCC + 1)]
    cols += [f"delta_{i}_std" for i in range(1, N_MFCC + 1)]
    cols += [f"delta2_{i}_mean_abs" for i in range(1, N_MFCC + 1)]
    cols += [f"delta2_{i}_std" for i in range(1, N_MFCC + 1)]
    return cols


def get_feature_group(feature_name: str) -> str:
    """Map a feature column name to its high-level group (Bộ A / MFCC / Delta / Delta-Delta)."""
    if feature_name.startswith("mfcc_"):
        return "Bộ B - MFCC (26D)"
    if feature_name.startswith("delta2_"):
        return "Delta-Delta MFCC (26D)"
    if feature_name.startswith("delta_"):
        return "Delta MFCC (26D)"
    return "Bộ A - Cơ bản (14D)"


def _inspect_original_audio_props(file_path: Path) -> tuple[int, int]:
    """Read original sample rate and channel count without resampling."""
    try:
        info = sf.info(str(file_path))
        return int(info.samplerate), int(info.channels)
    except Exception:
        # Fallback for MP3 or formats not handled directly by libsndfile
        y_orig, sr_orig = librosa.load(str(file_path), sr=None, mono=False)
        channels = 1 if y_orig.ndim == 1 else int(y_orig.shape[0])
        return int(sr_orig), channels


def compute_92d_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Compute the 92D Feature Set C vector from a 1D mono signal `y` at `sr` Hz.

    Returns:
        (vec_C of shape (92,), num_stft_frames)
    """
    if y is None or len(y) == 0:
        raise ValueError("Tệp âm thanh rỗng (0 mẫu).")
    if len(y) < MIN_SAMPLES:
        raise ValueError(
            f"Tệp âm thanh quá ngắn ({len(y)} mẫu < tối thiểu {MIN_SAMPLES} mẫu = 10 ms ở {SAMPLE_RATE} Hz)."
        )

    # 1. STFT (Hann window, n_fft=512, win_length=400, hop_length=160)
    stft_matrix = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
    )
    S = np.abs(stft_matrix)
    P = S ** 2
    num_stft_frames = int(S.shape[1])

    # 2. Set A: 7 basic descriptors -> 14D (mean, std)
    rms = librosa.feature.rms(y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=N_FFT)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=N_FFT)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=N_FFT, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S, power=2.0)[0]

    S_norm = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-10)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1, prepend=S_norm[:, :1]) ** 2, axis=0))

    vec_A: list[float] = []
    for feat in (rms, zcr, centroid, bandwidth, rolloff, flatness, flux):
        vec_A.extend([float(np.mean(feat)), float(np.std(feat))])

    # 3. Set B: MFCC (26D) + Delta (26D) + Delta-Delta (26D) = 78D
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(P), sr=sr, n_mfcc=N_MFCC)
    n_frames = mfcc.shape[1]

    if n_frames < 3:
        # Edge case for very short valid files (10–20 ms): edge-pad for delta calculation
        mfcc_for_delta = np.pad(mfcc, ((0, 0), (0, 3 - n_frames)), mode="edge")
        actual_width = 3
    else:
        mfcc_for_delta = mfcc
        actual_width = min(DELTA_WIDTH, n_frames if n_frames % 2 == 1 else n_frames - 1)
        actual_width = max(3, actual_width)

    delta_mfcc = librosa.feature.delta(mfcc_for_delta, width=actual_width, order=1)[:, :n_frames]
    delta2_mfcc = librosa.feature.delta(mfcc_for_delta, width=actual_width, order=2)[:, :n_frames]

    vec_C = np.concatenate([
        np.array(vec_A, dtype=np.float64),
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(np.abs(delta_mfcc), axis=1),
        np.std(delta_mfcc, axis=1),
        np.mean(np.abs(delta2_mfcc), axis=1),
        np.std(delta2_mfcc, axis=1),
    ]).astype(np.float64)

    if vec_C.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Số chiều đặc trưng không hợp lệ: {vec_C.shape}, kỳ vọng ({FEATURE_DIM},).")

    vec_C = np.nan_to_num(vec_C, nan=0.0, posinf=0.0, neginf=0.0)
    return vec_C, num_stft_frames


def extract_features_from_path(
    file_path: Path | str,
    filename: str | None = None,
) -> FeatureExtractionResult:
    """Load an audio file from disk, resample to 16 kHz mono, and extract its 92D vector."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp âm thanh: {path}")

    t0 = time.perf_counter()
    orig_sr, orig_channels = _inspect_original_audio_props(path)
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    vec_raw, num_stft_frames = compute_92d_vector(y, sr=sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    display_name = filename or path.name
    ext = Path(display_name).suffix.lstrip(".").upper() or "UNKNOWN"
    num_samples = int(len(y))
    duration_sec = float(num_samples / float(sr))
    num_frames_hop = int(np.ceil(num_samples / float(HOP_LENGTH)))

    metadata = AudioMetadata(
        filename=display_name,
        file_format=ext,
        file_size_bytes=int(path.stat().st_size),
        original_sample_rate=orig_sr,
        resampled_sample_rate=int(sr),
        original_channels=orig_channels,
        duration_sec=duration_sec,
        num_samples=num_samples,
        num_frames_hop=num_frames_hop,
        num_stft_frames=num_stft_frames,
        extraction_time_ms=elapsed_ms,
    )

    return FeatureExtractionResult(
        vector_raw=vec_raw,
        feature_names=build_feature_columns(),
        metadata=metadata,
        waveform=y.astype(np.float32),
    )


def extract_features_from_bytes(audio_bytes: bytes, filename: str) -> FeatureExtractionResult:
    """Extract 92D features and metadata from uploaded audio bytes."""
    if not audio_bytes:
        raise ValueError("Dữ liệu âm thanh đầu vào rỗng.")

    suffix = Path(filename).suffix.lower() or ".wav"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = Path(tmp_file.name)
        return extract_features_from_path(tmp_path, filename=filename)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
