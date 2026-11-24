"""KNN similarity search engine supporting Full-Scan, Cluster-Restricted, Multi-Cluster, and PCA strategies."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from web_app.config import (
    KMEANS_INIT,
    KMEANS_N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    KNN_ALGORITHM,
    KNN_DEFAULT_TOP_K,
    KNN_INDEX_PATH,
    KNN_LEAF_SIZE,
    KNN_METRIC,
    PCA_N_COMPONENTS,
    resolve_audio_path,
)
from web_app.engine.kmeans_engine import KMeansEngine, KMeansPrediction


@dataclass
class NeighborMatch:
    """One retrieved audio file in the Top-K similarity results."""

    rank: int
    dataset_index: int
    name: str
    species: str
    csv_path: str
    cluster_id: int
    distance: float
    duration: float
    frame_rate: int
    size: int
    audio_path: Path | None


@dataclass
class KNNSearchResult:
    """Structured result of a single KNN retrieval run."""

    mode_name: str
    algorithm: str
    metric: str
    n_neighbors: int
    candidates_scanned: int
    clusters_searched: list[int]
    fallback_used: bool
    kmeans_predict_time_ms: float
    search_time_ms: float
    total_retrieval_time_ms: float
    matches: list[NeighborMatch]
    predicted_species_majority: str
    predicted_species_weighted: str
    species_votes_weighted: dict[str, float]


def _vote_species(matches: list[NeighborMatch]) -> tuple[str, str, dict[str, float]]:
    """Compute majority vote and inverse-distance weighted vote for species prediction."""
    if not matches:
        return "Unknown", "Unknown", {}

    # Majority vote (tie-broken by closest match order)
    counts = Counter(m.species for m in matches)
    majority_species = max(
        counts.keys(),
        key=lambda sp: (counts[sp], -min(m.distance for m in matches if m.species == sp)),
    )

    # Inverse-distance weighted vote (Section 8.2)
    weights: dict[str, float] = defaultdict(float)
    for m in matches:
        weights[m.species] += 1.0 / (m.distance + 1e-6)
    total_w = sum(weights.values()) or 1.0
    norm_weights = {
        sp: round(w / total_w, 4)
        for sp, w in sorted(weights.items(), key=lambda item: -item[1])
    }
    weighted_species = next(iter(norm_weights.keys()))
    return majority_species, weighted_species, norm_weights


class KNNEngine:
    """Holds fitted KDTree indices (global, per-cluster, and PCA-reduced) for fast similarity search."""

    def __init__(
        self,
        X_zscore: np.ndarray,
        metadata_records: list[dict[str, Any]],
        cluster_labels: np.ndarray,
        global_kdtree: NearestNeighbors,
        global_balltree: NearestNeighbors,
        global_cosine: NearestNeighbors,
        cluster_kdtrees: dict[int, tuple[NearestNeighbors, np.ndarray]],
        pca_pipeline: dict[str, Any],
        global_fit_time_ms: float,
    ) -> None:
        self.X_zscore = np.asarray(X_zscore, dtype=np.float64)
        self.metadata_records = metadata_records
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        self.global_kdtree = global_kdtree
        self.global_balltree = global_balltree
        self.global_cosine = global_cosine
        self.cluster_kdtrees = cluster_kdtrees
        self.pca_pipeline = pca_pipeline
        self.global_fit_time_ms = float(global_fit_time_ms)

    @classmethod
    def fit(
        cls,
        X_zscore: np.ndarray,
        metadata_df: pd.DataFrame,
        kmeans_engine: KMeansEngine,
        leaf_size: int = KNN_LEAF_SIZE,
        pca_components: int = PCA_N_COMPONENTS,
    ) -> KNNEngine:
        """Fit global KDTree, per-cluster KDTrees, Cosine index, and PCA+KMeans+KNN pipeline."""
        X = np.asarray(X_zscore, dtype=np.float64)
        cluster_labels = kmeans_engine.cluster_labels

        # 1. Global KDTree (primary baseline without K-Means)
        t0 = time.perf_counter()
        global_kdtree = NearestNeighbors(
            algorithm=KNN_ALGORITHM,
            metric=KNN_METRIC,
            leaf_size=leaf_size,
        )
        global_kdtree.fit(X)
        global_fit_time_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Global BallTree & Cosine indices (for algorithm/metric comparison)
        global_balltree = NearestNeighbors(
            algorithm="ball_tree",
            metric="euclidean",
            leaf_size=leaf_size,
        ).fit(X)

        global_cosine = NearestNeighbors(
            algorithm="brute",
            metric="cosine",
        ).fit(X)

        # 3. Per-cluster KDTree indices (Section 8.2: KDTree inside each cluster)
        cluster_kdtrees: dict[int, tuple[NearestNeighbors, np.ndarray]] = {}
        for c_id, member_indices in kmeans_engine.cluster_members.items():
            if len(member_indices) > 0:
                nn = NearestNeighbors(
                    algorithm="kd_tree",
                    metric="euclidean",
                    leaf_size=min(leaf_size, max(1, len(member_indices))),
                )
                nn.fit(X[member_indices])
                cluster_kdtrees[c_id] = (nn, member_indices)

        # 4. Strategy 4 (Section 8.4): PCA(92 -> 20) + StandardScaler + KMeans(K=13) + per-cluster KDTree
        n_comp = min(pca_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=KMEANS_RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        pca_scaler = StandardScaler()
        X_pca_scaled = pca_scaler.fit_transform(X_pca)

        pca_kmeans = KMeans(
            n_clusters=KMEANS_N_CLUSTERS,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
        )
        pca_labels = pca_kmeans.fit_predict(X_pca_scaled)
        pca_cluster_kdtrees: dict[int, tuple[NearestNeighbors, np.ndarray]] = {}
        for c_id in range(KMEANS_N_CLUSTERS):
            idx = np.where(pca_labels == c_id)[0]
            if len(idx) > 0:
                nn_pca = NearestNeighbors(algorithm="kd_tree", metric="euclidean")
                nn_pca.fit(X_pca_scaled[idx])
                pca_cluster_kdtrees[c_id] = (nn_pca, idx)

        pca_pipeline = {
            "pca": pca,
            "scaler": pca_scaler,
            "kmeans": pca_kmeans,
            "X_pca_scaled": X_pca_scaled,
            "cluster_labels": pca_labels,
            "cluster_kdtrees": pca_cluster_kdtrees,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }

        records = metadata_df[
            ["name", "path", "species", "duration", "frame_rate", "size"]
        ].to_dict(orient="records")

        return cls(
            X_zscore=X,
            metadata_records=records,
            cluster_labels=cluster_labels,
            global_kdtree=global_kdtree,
            global_balltree=global_balltree,
            global_cosine=global_cosine,
            cluster_kdtrees=cluster_kdtrees,
            pca_pipeline=pca_pipeline,
            global_fit_time_ms=global_fit_time_ms,
        )

    def _build_matches(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        cluster_labels_override: np.ndarray | None = None,
    ) -> list[NeighborMatch]:
        labels = self.cluster_labels if cluster_labels_override is None else cluster_labels_override
        matches: list[NeighborMatch] = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
            idx_int = int(idx)
            rec = self.metadata_records[idx_int]
            matches.append(
                NeighborMatch(
                    rank=rank,
                    dataset_index=idx_int,
                    name=str(rec["name"]),
                    species=str(rec["species"]),
                    csv_path=str(rec["path"]),
                    cluster_id=int(labels[idx_int]),
                    distance=float(dist),
                    duration=float(rec["duration"]),
                    frame_rate=int(rec["frame_rate"]),
                    size=int(rec["size"]),
                    audio_path=resolve_audio_path(str(rec["path"])),
                )
            )
        return matches

    def search_full_scan(
        self,
        query_zscore: np.ndarray,
        top_k: int = KNN_DEFAULT_TOP_K,
        metric: str = "euclidean",
        algorithm: str = "kd_tree",
    ) -> KNNSearchResult:
        """Version 1 (No K-Means): Global KNN search across all 648 files."""
        q = np.asarray(query_zscore, dtype=np.float64).reshape(1, -1)
        k = min(int(top_k), len(self.X_zscore))

        t0 = time.perf_counter()
        if metric == "cosine":
            dists, idxs = self.global_cosine.kneighbors(q, n_neighbors=k)
            used_algo = "brute"
        elif algorithm == "ball_tree":
            dists, idxs = self.global_balltree.kneighbors(q, n_neighbors=k)
            used_algo = "ball_tree"
        elif algorithm == "brute":
            diffs = self.X_zscore - q[0]
            all_d = np.linalg.norm(diffs, axis=1)
            order = np.argsort(all_d)[:k]
            dists = all_d[order][None, :]
            idxs = order[None, :]
            used_algo = "brute"
        else:
            dists, idxs = self.global_kdtree.kneighbors(q, n_neighbors=k)
            used_algo = "kd_tree"
        search_ms = (time.perf_counter() - t0) * 1000.0

        matches = self._build_matches(idxs[0], dists[0])
        maj_sp, wt_sp, votes = _vote_species(matches)

        return KNNSearchResult(
            mode_name="Không K-means (Full-scan toàn cục)",
            algorithm=used_algo,
            metric=metric,
            n_neighbors=k,
            candidates_scanned=len(self.X_zscore),
            clusters_searched=list(range(len(self.cluster_kdtrees))),
            fallback_used=False,
            kmeans_predict_time_ms=0.0,
            search_time_ms=search_ms,
            total_retrieval_time_ms=search_ms,
            matches=matches,
            predicted_species_majority=maj_sp,
            predicted_species_weighted=wt_sp,
            species_votes_weighted=votes,
        )

    def search_with_kmeans(
        self,
        query_zscore: np.ndarray,
        kmeans_engine: KMeansEngine,
        top_k: int = KNN_DEFAULT_TOP_K,
        n_probe_clusters: int = 1,
        metric: str = "euclidean",
        prediction: KMeansPrediction | None = None,
    ) -> tuple[KNNSearchResult, KMeansPrediction]:
        """Version 2 (With K-Means): Predict cluster(s) and run KNN only on cluster members.

        Includes automatic fallback to adjacent clusters if the selected cluster(s)
        contain fewer than `top_k` files (Section 8.3).
        """
        q = np.asarray(query_zscore, dtype=np.float64).reshape(1, -1)
        pred = prediction if prediction is not None else kmeans_engine.predict(q[0])
        k = min(int(top_k), len(self.X_zscore))
        n_probe = max(1, min(int(n_probe_clusters), pred.n_clusters))

        t0 = time.perf_counter()

        # Select initial target clusters
        selected_clusters = list(pred.nearest_clusters_order[:n_probe])
        fallback_used = False

        # Fast path: single cluster, Euclidean, pre-fitted per-cluster KDTree, enough members
        if (
            n_probe == 1
            and metric == "euclidean"
            and selected_clusters[0] in self.cluster_kdtrees
            and len(self.cluster_kdtrees[selected_clusters[0]][1]) >= k
        ):
            nn_cluster, member_indices = self.cluster_kdtrees[selected_clusters[0]]
            dists, local_idxs = nn_cluster.kneighbors(q, n_neighbors=k)
            global_idxs = member_indices[local_idxs[0]]
            top_dists = dists[0]
            candidates_scanned = int(len(member_indices))
            used_algo = "kd_tree (per-cluster)"
        else:
            # Multi-cluster probe, Cosine metric, or Fallback when cluster size < top_k
            candidate_parts = [kmeans_engine.get_cluster_indices(c) for c in selected_clusters]
            candidate_indices = np.concatenate(candidate_parts) if candidate_parts else np.array([], dtype=np.int64)

            next_pos = n_probe
            while len(candidate_indices) < k and next_pos < len(pred.nearest_clusters_order):
                fallback_used = True
                next_c = pred.nearest_clusters_order[next_pos]
                selected_clusters.append(next_c)
                extra = kmeans_engine.get_cluster_indices(next_c)
                if len(extra) > 0:
                    candidate_indices = np.concatenate([candidate_indices, extra])
                next_pos += 1

            sub_X = self.X_zscore[candidate_indices]
            candidates_scanned = int(len(candidate_indices))

            if metric == "cosine":
                q_norm = np.linalg.norm(q[0]) + 1e-12
                sub_norms = np.linalg.norm(sub_X, axis=1) + 1e-12
                all_d = 1.0 - (sub_X @ q[0]) / (sub_norms * q_norm)
                used_algo = "cosine (cluster subset)"
            else:
                all_d = np.linalg.norm(sub_X - q[0], axis=1)
                used_algo = "euclidean (multi-cluster)" if n_probe > 1 else "euclidean (cluster + fallback)"

            order = np.argsort(all_d)[:k]
            global_idxs = candidate_indices[order]
            top_dists = all_d[order]

        search_ms = (time.perf_counter() - t0) * 1000.0
        matches = self._build_matches(global_idxs, top_dists)
        maj_sp, wt_sp, votes = _vote_species(matches)

        mode_label = (
            f"Có K-means (K={pred.n_clusters}, KNN trong cụm)"
            if n_probe == 1
            else f"Có K-means (K={pred.n_clusters}, Top-{n_probe} cụm)"
        )

        result = KNNSearchResult(
            mode_name=mode_label,
            algorithm=used_algo,
            metric=metric,
            n_neighbors=k,
            candidates_scanned=candidates_scanned,
            clusters_searched=selected_clusters,
            fallback_used=fallback_used,
            kmeans_predict_time_ms=pred.predict_time_ms,
            search_time_ms=search_ms,
            total_retrieval_time_ms=pred.predict_time_ms + search_ms,
            matches=matches,
            predicted_species_majority=maj_sp,
            predicted_species_weighted=wt_sp,
            species_votes_weighted=votes,
        )
        return result, pred

    def search_pca_kmeans(
        self,
        query_zscore: np.ndarray,
        top_k: int = KNN_DEFAULT_TOP_K,
        n_probe_clusters: int = 1,
    ) -> KNNSearchResult:
        """Strategy 4 (Section 8.4): PCA(92->20) + StandardScaler + K-Means + KNN."""
        q = np.asarray(query_zscore, dtype=np.float64).reshape(1, -1)
        k = min(int(top_k), len(self.X_zscore))

        pca: PCA = self.pca_pipeline["pca"]
        pca_scaler: StandardScaler = self.pca_pipeline["scaler"]
        pca_km: KMeans = self.pca_pipeline["kmeans"]
        X_pca_scaled: np.ndarray = self.pca_pipeline["X_pca_scaled"]
        pca_labels: np.ndarray = self.pca_pipeline["cluster_labels"]
        pca_kdtrees: dict[int, tuple[NearestNeighbors, np.ndarray]] = self.pca_pipeline["cluster_kdtrees"]

        t_km0 = time.perf_counter()
        q_pca = pca_scaler.transform(pca.transform(q))
        diffs_c = pca_km.cluster_centers_ - q_pca[0]
        centroid_dists = np.linalg.norm(diffs_c, axis=1)
        nearest_order = [int(i) for i in np.argsort(centroid_dists)]
        km_ms = (time.perf_counter() - t_km0) * 1000.0

        t_s0 = time.perf_counter()
        n_probe = max(1, min(int(n_probe_clusters), pca_km.n_clusters))
        selected = nearest_order[:n_probe]
        fallback_used = False

        if n_probe == 1 and selected[0] in pca_kdtrees and len(pca_kdtrees[selected[0]][1]) >= k:
            nn_c, member_idx = pca_kdtrees[selected[0]]
            dists, local_idx = nn_c.kneighbors(q_pca, n_neighbors=k)
            global_idxs = member_idx[local_idx[0]]
            top_dists = dists[0]
            candidates_scanned = int(len(member_idx))
        else:
            cand_parts = [np.where(pca_labels == c)[0] for c in selected]
            cand_idx = np.concatenate(cand_parts)
            pos = n_probe
            while len(cand_idx) < k and pos < len(nearest_order):
                fallback_used = True
                nxt = nearest_order[pos]
                selected.append(nxt)
                cand_idx = np.concatenate([cand_idx, np.where(pca_labels == nxt)[0]])
                pos += 1
            sub_X = X_pca_scaled[cand_idx]
            all_d = np.linalg.norm(sub_X - q_pca[0], axis=1)
            order = np.argsort(all_d)[:k]
            global_idxs = cand_idx[order]
            top_dists = all_d[order]
            candidates_scanned = int(len(cand_idx))

        search_ms = (time.perf_counter() - t_s0) * 1000.0
        matches = self._build_matches(global_idxs, top_dists, cluster_labels_override=pca_labels)
        maj_sp, wt_sp, votes = _vote_species(matches)

        return KNNSearchResult(
            mode_name=f"PCA(92→{pca.n_components_}) + K-Means(K={pca_km.n_clusters}) + KNN",
            algorithm="kd_tree (PCA space)",
            metric="euclidean",
            n_neighbors=k,
            candidates_scanned=candidates_scanned,
            clusters_searched=selected,
            fallback_used=fallback_used,
            kmeans_predict_time_ms=km_ms,
            search_time_ms=search_ms,
            total_retrieval_time_ms=km_ms + search_ms,
            matches=matches,
            predicted_species_majority=maj_sp,
            predicted_species_weighted=wt_sp,
            species_votes_weighted=votes,
        )

    def save(self, path: Path | str = KNN_INDEX_PATH) -> Path:
        """Persist fitted KNN indices with joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "X_zscore": self.X_zscore,
            "metadata_records": self.metadata_records,
            "cluster_labels": self.cluster_labels,
            "global_kdtree": self.global_kdtree,
            "global_balltree": self.global_balltree,
            "global_cosine": self.global_cosine,
            "cluster_kdtrees": self.cluster_kdtrees,
            "pca_pipeline": self.pca_pipeline,
            "global_fit_time_ms": self.global_fit_time_ms,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str = KNN_INDEX_PATH) -> KNNEngine:
        """Load persisted KNN index from joblib."""
        in_path = Path(path)
        if not in_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy chỉ mục KNN: {in_path}. Hãy chạy `python scripts/train_models.py` trước."
            )
        payload = joblib.load(in_path)
        return cls(**payload)
