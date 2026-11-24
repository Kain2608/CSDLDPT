"""K-Means clustering engine (K=13, k-means++, n_init=10) with evaluation metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from web_app.config import (
    KMEANS_INIT,
    KMEANS_MODEL_PATH,
    KMEANS_N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
)


@dataclass
class KMeansPrediction:
    """Result of predicting the nearest cluster for a 92D Z-score query vector."""

    n_clusters: int
    cluster_id: int
    centroid: np.ndarray
    distance_to_centroid: float
    all_centroid_distances: np.ndarray
    nearest_clusters_order: list[int]
    cluster_size: int
    species_distribution: dict[str, int]
    predict_time_ms: float


def compute_purity(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    """Compute clustering purity against ground-truth species labels."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)
    if n_samples == 0:
        return 0.0

    correct = 0
    for cluster_id in np.unique(y_pred_arr):
        mask = y_pred_arr == cluster_id
        _, counts = np.unique(y_true_arr[mask], return_counts=True)
        correct += int(np.max(counts))
    return float(correct / n_samples)


def evaluate_clustering(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    y_true: np.ndarray | pd.Series | None = None,
    inertia: float | None = None,
) -> dict[str, float]:
    """Compute internal & external clustering quality metrics."""
    metrics: dict[str, float] = {
        "silhouette": float(silhouette_score(X, cluster_labels)),
        "davies_bouldin": float(davies_bouldin_score(X, cluster_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, cluster_labels)),
    }
    if inertia is not None:
        metrics["inertia"] = float(inertia)
    if y_true is not None:
        metrics["ari"] = float(adjusted_rand_score(y_true, cluster_labels))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, cluster_labels))
        metrics["purity"] = compute_purity(y_true, cluster_labels)
    return metrics


def sweep_kmeans_k(
    X: np.ndarray,
    y_true: np.ndarray | pd.Series,
    k_values: list[int],
    n_init: int = KMEANS_N_INIT,
    random_state: int = KMEANS_RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Run K-Means across a range of K values to compute Elbow & clustering quality curves."""
    results: list[dict[str, Any]] = []
    for k in k_values:
        t0 = time.perf_counter()
        km = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            n_init=n_init,
            random_state=random_state,
        )
        labels = km.fit_predict(X)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        metrics = evaluate_clustering(X, labels, y_true=y_true, inertia=float(km.inertia_))
        results.append({
            "k": int(k),
            "fit_time_ms": round(fit_ms, 3),
            **{key: round(val, 6) for key, val in metrics.items()},
        })
    return results


class KMeansEngine:
    """Manages offline K-Means training, persistence, and online cluster prediction."""

    def __init__(
        self,
        model: KMeans,
        cluster_labels: np.ndarray,
        species_list: list[str],
        metrics: dict[str, float],
        fit_time_ms: float,
    ) -> None:
        self.model = model
        self.n_clusters = int(model.n_clusters)
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        self.species_list = list(species_list)
        self.metrics = dict(metrics)
        self.fit_time_ms = float(fit_time_ms)

        self.cluster_members: dict[int, np.ndarray] = {}
        self.cluster_species_dist: dict[int, dict[str, int]] = {}
        self._build_cluster_lookup()

    def _build_cluster_lookup(self) -> None:
        species_arr = np.asarray(self.species_list)
        for c in range(self.n_clusters):
            indices = np.where(self.cluster_labels == c)[0]
            self.cluster_members[c] = indices
            if len(indices) > 0:
                vals, counts = np.unique(species_arr[indices], return_counts=True)
                dist = {
                    str(v): int(cnt)
                    for v, cnt in sorted(zip(vals, counts), key=lambda item: (-item[1], item[0]))
                }
            else:
                dist = {}
            self.cluster_species_dist[c] = dist

    @classmethod
    def train(
        cls,
        X_zscore: np.ndarray,
        species_list: list[str] | pd.Series,
        n_clusters: int = KMEANS_N_CLUSTERS,
        init: str = KMEANS_INIT,
        n_init: int = KMEANS_N_INIT,
        random_state: int = KMEANS_RANDOM_STATE,
    ) -> KMeansEngine:
        """Train K-Means offline on the normalized 648 x 92 feature matrix."""
        X = np.asarray(X_zscore, dtype=np.float64)
        species = [str(s) for s in species_list]

        t0 = time.perf_counter()
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        metrics = evaluate_clustering(X, labels, y_true=np.asarray(species), inertia=float(model.inertia_))
        return cls(
            model=model,
            cluster_labels=labels,
            species_list=species,
            metrics=metrics,
            fit_time_ms=fit_time_ms,
        )

    def predict(self, query_zscore: np.ndarray) -> KMeansPrediction:
        """Assign a 92D Z-score query vector to its nearest cluster and compute centroid distances."""
        q = np.asarray(query_zscore, dtype=np.float64).reshape(1, -1)

        t0 = time.perf_counter()
        cluster_id = int(self.model.predict(q)[0])
        # Compute Euclidean distances to all centroids for reporting & multi-cluster fallback
        diffs = self.model.cluster_centers_ - q[0]
        all_dists = np.linalg.norm(diffs, axis=1)
        nearest_order = [int(i) for i in np.argsort(all_dists)]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return KMeansPrediction(
            n_clusters=self.n_clusters,
            cluster_id=cluster_id,
            centroid=self.model.cluster_centers_[cluster_id].copy(),
            distance_to_centroid=float(all_dists[cluster_id]),
            all_centroid_distances=all_dists,
            nearest_clusters_order=nearest_order,
            cluster_size=int(len(self.cluster_members[cluster_id])),
            species_distribution=dict(self.cluster_species_dist[cluster_id]),
            predict_time_ms=elapsed_ms,
        )

    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """Return dataset row indices belonging to `cluster_id`."""
        return self.cluster_members[int(cluster_id)]

    def save(self, path: Path | str = KMEANS_MODEL_PATH) -> Path:
        """Persist trained K-Means engine with joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "cluster_labels": self.cluster_labels,
            "species_list": self.species_list,
            "metrics": self.metrics,
            "fit_time_ms": self.fit_time_ms,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str = KMEANS_MODEL_PATH) -> KMeansEngine:
        """Load persisted K-Means engine from joblib."""
        in_path = Path(path)
        if not in_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình K-Means: {in_path}. Hãy chạy `python scripts/train_models.py` trước."
            )
        payload = joblib.load(in_path)
        return cls(
            model=payload["model"],
            cluster_labels=payload["cluster_labels"],
            species_list=payload["species_list"],
            metrics=payload["metrics"],
            fit_time_ms=payload["fit_time_ms"],
        )
