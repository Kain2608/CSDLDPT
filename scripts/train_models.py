"""Offline training pipeline: train K-Means (K=13), fit KDTree KNN indices, and save artifacts."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import prepare_data
from web_app.config import (
    KMEANS_INIT,
    KMEANS_MODEL_PATH,
    KMEANS_N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    KNN_ALGORITHM,
    KNN_INDEX_PATH,
    KNN_LEAF_SIZE,
    KNN_METRIC,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PCA_N_COMPONENTS,
    SCALER_PARAMS_JSON,
    ZSCORE_FEATURES_CSV,
)
from web_app.engine.kmeans_engine import KMeansEngine
from web_app.engine.knn_engine import KNNEngine
from web_app.engine.scaler import ZScoreScaler


def train_all() -> dict:
    """Run full offline model training and persist joblib/JSON/projection artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not ZSCORE_FEATURES_CSV.is_file() or not SCALER_PARAMS_JSON.is_file():
        prepare_data(force=True)

    # 1. Measure CSV + scaler load time (Section 3.5)
    t_load0 = time.perf_counter()
    df_zscore = pd.read_csv(ZSCORE_FEATURES_CSV)
    scaler = ZScoreScaler.from_json(SCALER_PARAMS_JSON)
    load_time_ms = (time.perf_counter() - t_load0) * 1000.0

    feature_names = scaler.feature_names
    X_zscore = df_zscore[feature_names].to_numpy(dtype=np.float64)
    species = df_zscore["species"].astype(str).tolist()

    # 2. Train K-Means (K=13, k-means++, n_init=10, random_state=42)
    print(f"[train_models] Training KMeans(K={KMEANS_N_CLUSTERS}, init={KMEANS_INIT}, n_init={KMEANS_N_INIT})...")
    kmeans_engine = KMeansEngine.train(
        X_zscore=X_zscore,
        species_list=species,
        n_clusters=KMEANS_N_CLUSTERS,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    kmeans_engine.save(KMEANS_MODEL_PATH)

    # 3. Fit Global KDTree + Per-Cluster KDTrees + PCA Pipeline
    print(f"[train_models] Fitting KNNEngine(algorithm={KNN_ALGORITHM}, metric={KNN_METRIC})...")
    knn_engine = KNNEngine.fit(
        X_zscore=X_zscore,
        metadata_df=df_zscore,
        kmeans_engine=kmeans_engine,
        leaf_size=KNN_LEAF_SIZE,
        pca_components=PCA_N_COMPONENTS,
    )
    knn_engine.save(KNN_INDEX_PATH)

    # 4. Precompute 2D PCA & 2D t-SNE projections for instant visualization in Streamlit
    print("[train_models] Computing 2D PCA and t-SNE projections for cluster visualization...")
    pca_2d = PCA(n_components=2, random_state=KMEANS_RANDOM_STATE)
    coords_pca = pca_2d.fit_transform(X_zscore)
    centroids_pca = pca_2d.transform(kmeans_engine.model.cluster_centers_)

    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=KMEANS_RANDOM_STATE,
    )
    coords_tsne = tsne_2d.fit_transform(X_zscore)

    proj_df = pd.DataFrame({
        "dataset_index": np.arange(len(df_zscore)),
        "name": df_zscore["name"].astype(str),
        "species": df_zscore["species"].astype(str),
        "path": df_zscore["path"].astype(str),
        "duration": df_zscore["duration"].astype(float),
        "cluster_id": kmeans_engine.cluster_labels.astype(int),
        "pca_x": coords_pca[:, 0],
        "pca_y": coords_pca[:, 1],
        "tsne_x": coords_tsne[:, 0],
        "tsne_y": coords_tsne[:, 1],
    })
    proj_path = MODELS_DIR / "projections_2d.csv"
    proj_df.to_csv(proj_path, index=False)

    centroids_df = pd.DataFrame({
        "cluster_id": np.arange(KMEANS_N_CLUSTERS),
        "pca_x": centroids_pca[:, 0],
        "pca_y": centroids_pca[:, 1],
        "size": [int(len(kmeans_engine.cluster_members[c])) for c in range(KMEANS_N_CLUSTERS)],
    })
    centroids_df.to_csv(MODELS_DIR / "centroids_pca_2d.csv", index=False)

    # 5. Build and persist model metadata JSON
    existing_meta: dict = {}
    if MODEL_METADATA_PATH.is_file():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)

    species_counts = df_zscore["species"].value_counts().sort_index().to_dict()
    cluster_summary = []
    for c in range(KMEANS_N_CLUSTERS):
        dist = kmeans_engine.cluster_species_dist[c]
        dominant = next(iter(dist.items())) if dist else ("None", 0)
        size = int(len(kmeans_engine.cluster_members[c]))
        cluster_summary.append({
            "cluster_id": c,
            "size": size,
            "dominant_species": dominant[0],
            "dominant_count": int(dominant[1]),
            "dominant_ratio": round(dominant[1] / max(1, size), 4),
            "species_distribution": dist,
        })

    metadata = {
        **existing_meta,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "n_files": int(len(df_zscore)),
            "n_species": int(df_zscore["species"].nunique()),
            "n_features": int(len(feature_names)),
            "species_counts": {str(k): int(v) for k, v in species_counts.items()},
        },
        "hyperparameters": {
            "kmeans_n_clusters": KMEANS_N_CLUSTERS,
            "kmeans_init": KMEANS_INIT,
            "kmeans_n_init": KMEANS_N_INIT,
            "kmeans_random_state": KMEANS_RANDOM_STATE,
            "knn_algorithm": KNN_ALGORITHM,
            "knn_metric": KNN_METRIC,
            "knn_leaf_size": KNN_LEAF_SIZE,
            "pca_n_components": PCA_N_COMPONENTS,
        },
        "init_timings_ms": {
            "load_csv_and_scaler_ms": round(load_time_ms, 3),
            "kmeans_fit_ms": round(kmeans_engine.fit_time_ms, 3),
            "kdtree_fit_ms": round(knn_engine.global_fit_time_ms, 3),
        },
        "kmeans_metrics": {k: round(v, 6) for k, v in kmeans_engine.metrics.items()},
        "pca_2d_explained_variance_ratio": [round(float(v), 4) for v in pca_2d.explained_variance_ratio_],
        "pca_strategy_explained_variance_sum": round(
            float(knn_engine.pca_pipeline["explained_variance_ratio_sum"]), 4
        ),
        "cluster_summary": cluster_summary,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"[train_models] Done! KMeans fit={kmeans_engine.fit_time_ms:.2f} ms, "
        f"KDTree fit={knn_engine.global_fit_time_ms:.2f} ms, "
        f"Silhouette={kmeans_engine.metrics['silhouette']:.4f}, "
        f"DB={kmeans_engine.metrics['davies_bouldin']:.4f}, "
        f"ARI={kmeans_engine.metrics['ari']:.4f}, "
        f"NMI={kmeans_engine.metrics['nmi']:.4f}, "
        f"Purity={kmeans_engine.metrics['purity']:.4f}"
    )
    return metadata


if __name__ == "__main__":
    train_all()
