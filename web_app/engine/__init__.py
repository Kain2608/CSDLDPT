"""Core AI engine modules: feature extraction, Z-score scaling, K-Means, KNN, and search pipeline."""

from web_app.engine.feature_engine import (
    AudioMetadata,
    FeatureExtractionResult,
    build_feature_columns,
    extract_features_from_bytes,
    extract_features_from_path,
    get_feature_group,
)
from web_app.engine.kmeans_engine import (
    KMeansEngine,
    KMeansPrediction,
    compute_purity,
    evaluate_clustering,
    sweep_kmeans_k,
)
from web_app.engine.knn_engine import KNNEngine, KNNSearchResult, NeighborMatch
from web_app.engine.pipeline import AnimalSoundPipeline, PipelineQueryResult
from web_app.engine.scaler import ScalerResult, ZScoreScaler

__all__ = [
    "AudioMetadata",
    "FeatureExtractionResult",
    "build_feature_columns",
    "extract_features_from_bytes",
    "extract_features_from_path",
    "get_feature_group",
    "ScalerResult",
    "ZScoreScaler",
    "KMeansEngine",
    "KMeansPrediction",
    "compute_purity",
    "evaluate_clustering",
    "sweep_kmeans_k",
    "KNNEngine",
    "KNNSearchResult",
    "NeighborMatch",
    "AnimalSoundPipeline",
    "PipelineQueryResult",
]
