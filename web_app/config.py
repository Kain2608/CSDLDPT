"""Centralized configuration for the Animal Sound K-Means + KNN AI project."""

from __future__ import annotations

import os
from pathlib import Path

# Directory layout
WEB_APP_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = WEB_APP_DIR.parent
SOURCE_OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DATA_DIR: Path = WEB_APP_DIR / "data"
MODELS_DIR: Path = WEB_APP_DIR / "models"
STATIC_DIR: Path = WEB_APP_DIR / "static"

# Dataset & scaler artifacts
RAW_FEATURES_CSV: Path = DATA_DIR / "features_raw_92d.csv"
ZSCORE_FEATURES_CSV: Path = DATA_DIR / "features_zscore_92d.csv"
SCALER_PARAMS_JSON: Path = DATA_DIR / "zscore_scaler_params.json"

# Persisted model artifacts
KMEANS_MODEL_PATH: Path = MODELS_DIR / "kmeans_model.joblib"
KNN_INDEX_PATH: Path = MODELS_DIR / "knn_index.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# Static report charts
ELBOW_SILHOUETTE_PNG: Path = STATIC_DIR / "elbow_silhouette.png"
PCA_SCATTER_PNG: Path = STATIC_DIR / "pca_scatter.png"
SPECIES_CLUSTER_HEATMAP_PNG: Path = STATIC_DIR / "species_cluster_heatmap.png"

# Audio DSP configuration (exact match with extract_features.py)
SAMPLE_RATE: int = 16000
WIN_LENGTH: int = 400       # 25 ms
HOP_LENGTH: int = 160       # 10 ms
N_FFT: int = 512
N_MFCC: int = 13
DELTA_WIDTH: int = 9        # K = 4
FEATURE_DIM: int = 92
MIN_SAMPLES: int = 160      # 10 ms minimum audio validation
SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = ("wav", "mp3", "flac", "ogg")

# Metadata columns stored in CSV files
METADATA_COLUMNS: tuple[str, ...] = (
    "name",
    "path",
    "species",
    "channels",
    "sample_width",
    "frame_rate",
    "nframes",
    "duration",
    "size",
)

# K-Means hyperparameters (Section 4.2 & 5.3)
KMEANS_N_CLUSTERS: int = 13
KMEANS_INIT: str = "k-means++"
KMEANS_N_INIT: int = 10
KMEANS_RANDOM_STATE: int = 42

# KNN hyperparameters (Section 3.4 & 5.4)
KNN_ALGORITHM: str = "kd_tree"
KNN_METRIC: str = "euclidean"
KNN_DEFAULT_TOP_K: int = 5
KNN_LEAF_SIZE: int = 30

# Advanced Strategy 4 (Section 8.4): PCA(92 -> 20) + K-Means + KNN
PCA_N_COMPONENTS: int = 20

# Benchmark parameters (Section 3.3, 3.4, 4.3)
BENCHMARK_N_QUERIES: int = 200
BENCHMARK_RANDOM_STATE: int = 42
K_SWEEP_RANGE: list[int] = list(range(2, 21))
SPEED_BENCHMARK_KS: list[int] = [4, 8, 10, 13, 16, 20, 26]


def get_audio_search_roots() -> list[Path]:
    """Return ordered candidate directories where dataset WAV files may reside."""
    roots: list[Path] = []
    env_dir = os.environ.get("ANIMAL_SOUND_AUDIO_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.extend([
        PROJECT_ROOT / "Animal-Soundprepros",
        DATA_DIR / "audio",
        PROJECT_ROOT,
        Path("C:/HCSDLDPT/Animal-Soundprepros"),
        Path("C:/HCSDLDPT"),
    ])
    return roots


def resolve_audio_path(relative_path: str) -> Path | None:
    """Resolve a dataset CSV `path` (e.g. 'Animal-Soundprepros\\Bear\\Bear_1.wav') to an existing WAV file."""
    if not relative_path:
        return None

    normalized = relative_path.replace("\\", "/")
    rel = Path(normalized)

    # Also build path stripped of leading 'Animal-Soundprepros/' folder if present
    parts = rel.parts
    stripped_rel = Path(*parts[1:]) if len(parts) > 1 and parts[0].lower() == "animal-soundprepros" else rel

    for root in get_audio_search_roots():
        for candidate in (root / rel, root / stripped_rel):
            if candidate.is_file():
                return candidate
    return None
