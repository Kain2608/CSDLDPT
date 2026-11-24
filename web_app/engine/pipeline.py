"""End-to-end audio similarity search pipeline orchestrating extraction, Z-score, K-Means, and KNN."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from web_app.config import (
    KMEANS_MODEL_PATH,
    KNN_DEFAULT_TOP_K,
    KNN_INDEX_PATH,
    MODEL_METADATA_PATH,
    RAW_FEATURES_CSV,
    SCALER_PARAMS_JSON,
    ZSCORE_FEATURES_CSV,
    resolve_audio_path,
)
from web_app.engine.feature_engine import (
    AudioMetadata,
    FeatureExtractionResult,
    extract_features_from_bytes,
    extract_features_from_path,
)
from web_app.engine.kmeans_engine import KMeansEngine, KMeansPrediction
from web_app.engine.knn_engine import KNNEngine, KNNSearchResult
from web_app.engine.scaler import ScalerResult, ZScoreScaler
from web_app.engine.pgvector_repository import PgVectorRepository


@dataclass
class PipelineQueryResult:
    """Complete end-to-end search output with full technical diagnostics for the web app."""

    extraction: FeatureExtractionResult
    scaling: ScalerResult
    kmeans_prediction: KMeansPrediction
    result_without_kmeans: KNNSearchResult | None
    result_with_kmeans: KNNSearchResult | None
    result_multi_cluster: KNNSearchResult | None
    result_pca_kmeans: KNNSearchResult | None
    comparison_summary: dict[str, Any] | None
    total_end_to_end_ms: float


class AnimalSoundPipeline:
    """Loads persisted models & dataset artifacts and serves audio similarity queries."""

    def __init__(
        self,
        scaler: ZScoreScaler,
        kmeans_engine: KMeansEngine,
        knn_engine: KNNEngine,
        df_zscore: pd.DataFrame,
        df_raw: pd.DataFrame,
        model_metadata: dict[str, Any],
    ) -> None:
        self.scaler = scaler
        self.kmeans_engine = kmeans_engine
        self.knn_engine = knn_engine
        self.df_zscore = df_zscore
        self.df_raw = df_raw
        self.model_metadata = model_metadata
        self.pg_repo = PgVectorRepository()

    @classmethod
    def load(
        cls,
        scaler_path: Path | str = SCALER_PARAMS_JSON,
        kmeans_path: Path | str = KMEANS_MODEL_PATH,
        knn_path: Path | str = KNN_INDEX_PATH,
        zscore_csv: Path | str = ZSCORE_FEATURES_CSV,
        raw_csv: Path | str = RAW_FEATURES_CSV,
        metadata_json: Path | str = MODEL_METADATA_PATH,
    ) -> AnimalSoundPipeline:
        """Load all persisted inference artifacts."""
        scaler = ZScoreScaler.from_json(scaler_path)
        kmeans_engine = KMeansEngine.load(kmeans_path)
        knn_engine = KNNEngine.load(knn_path)
        df_zscore = pd.read_csv(zscore_csv)
        df_raw = pd.read_csv(raw_csv)
        df_zscore["cluster_id"] = kmeans_engine.cluster_labels
        df_raw["cluster_id"] = kmeans_engine.cluster_labels

        meta_path = Path(metadata_json)
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {}

        return cls(
            scaler=scaler,
            kmeans_engine=kmeans_engine,
            knn_engine=knn_engine,
            df_zscore=df_zscore,
            df_raw=df_raw,
            model_metadata=model_metadata,
        )

    def _run_retrieval(
        self,
        extraction: FeatureExtractionResult,
        mode: str = "compare",
        top_k: int = KNN_DEFAULT_TOP_K,
        metric: str = "euclidean",
        algorithm: str = "kd_tree",
        n_probe_clusters: int = 1,
        include_advanced_strategies: bool = False,
        backend: str = "kdtree",
        t_start: float | None = None,
    ) -> PipelineQueryResult:
        t0 = t_start if t_start is not None else time.perf_counter()

        # Step 2: Frozen Z-score normalization
        scaling = self.scaler.transform(extraction.vector_raw)
        z_vec = scaling.vector_zscore

        res_no_km: KNNSearchResult | None = None
        res_with_km: KNNSearchResult | None = None
        res_multi: KNNSearchResult | None = None
        res_pca: KNNSearchResult | None = None
        comparison: dict[str, Any] | None = None

        if backend == "pgvector":
            km_pred = self.pg_repo.predict_cluster_sql(z_vec)
            if mode in ("without_kmeans", "compare"):
                res_no_km = self.pg_repo.search_full_scan_sql(
                    z_vec,
                    top_k=top_k,
                    metric=metric,
                )
            if mode in ("with_kmeans", "compare"):
                res_with_km, km_pred = self.pg_repo.search_with_kmeans_sql(
                    z_vec,
                    top_k=top_k,
                    n_probe_clusters=n_probe_clusters,
                    metric=metric,
                    prediction=km_pred,
                )
        else:
            # Always compute K-Means cluster assignment for technical diagnostics
            km_pred = self.kmeans_engine.predict(z_vec)

            if mode in ("without_kmeans", "compare"):
                res_no_km = self.knn_engine.search_full_scan(
                    z_vec,
                    top_k=top_k,
                    metric=metric,
                    algorithm=algorithm,
                )

            if mode in ("with_kmeans", "compare"):
                res_with_km, km_pred = self.knn_engine.search_with_kmeans(
                    z_vec,
                    self.kmeans_engine,
                    top_k=top_k,
                    n_probe_clusters=n_probe_clusters,
                    metric=metric,
                    prediction=km_pred,
                )

        if include_advanced_strategies:
            res_multi, _ = self.knn_engine.search_with_kmeans(
                z_vec,
                self.kmeans_engine,
                top_k=top_k,
                n_probe_clusters=2,
                metric=metric,
                prediction=km_pred,
            )
            res_pca = self.knn_engine.search_pca_kmeans(z_vec, top_k=top_k, n_probe_clusters=1)

        if res_no_km is not None and res_with_km is not None:
            set_full = {m.dataset_index for m in res_no_km.matches}
            set_km = {m.dataset_index for m in res_with_km.matches}
            overlap_count = len(set_full & set_km)
            denom = max(res_with_km.search_time_ms, 1e-6)
            speedup_knn_only = res_no_km.search_time_ms / denom
            denom_total = max(res_with_km.total_retrieval_time_ms, 1e-6)
            speedup_total = res_no_km.total_retrieval_time_ms / denom_total

            comparison = {
                "full_search_ms": res_no_km.search_time_ms,
                "kmeans_search_ms": res_with_km.search_time_ms,
                "kmeans_total_retrieval_ms": res_with_km.total_retrieval_time_ms,
                "full_candidates": res_no_km.candidates_scanned,
                "kmeans_candidates": res_with_km.candidates_scanned,
                "candidate_reduction_pct": round(
                    (1.0 - res_with_km.candidates_scanned / max(1, res_no_km.candidates_scanned)) * 100.0,
                    2,
                ),
                "speedup_knn_only": round(speedup_knn_only, 2),
                "speedup_total_retrieval": round(speedup_total, 2),
                "overlap_count": overlap_count,
                "overlap_ratio": round(overlap_count / max(1, top_k), 4),
                "top1_match_same": (
                    bool(res_no_km.matches and res_with_km.matches)
                    and res_no_km.matches[0].dataset_index == res_with_km.matches[0].dataset_index
                ),
            }

        total_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineQueryResult(
            extraction=extraction,
            scaling=scaling,
            kmeans_prediction=km_pred,
            result_without_kmeans=res_no_km,
            result_with_kmeans=res_with_km,
            result_multi_cluster=res_multi,
            result_pca_kmeans=res_pca,
            comparison_summary=comparison,
            total_end_to_end_ms=total_ms,
        )

    def query_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        mode: str = "compare",
        top_k: int = KNN_DEFAULT_TOP_K,
        metric: str = "euclidean",
        algorithm: str = "kd_tree",
        n_probe_clusters: int = 1,
        include_advanced_strategies: bool = False,
        backend: str = "kdtree",
    ) -> PipelineQueryResult:
        """Run the full pipeline starting from uploaded file bytes."""
        t0 = time.perf_counter()
        extraction = extract_features_from_bytes(audio_bytes, filename)
        return self._run_retrieval(
            extraction=extraction,
            mode=mode,
            top_k=top_k,
            metric=metric,
            algorithm=algorithm,
            n_probe_clusters=n_probe_clusters,
            include_advanced_strategies=include_advanced_strategies,
            backend=backend,
            t_start=t0,
        )

    def query_audio_path(
        self,
        file_path: Path | str,
        mode: str = "compare",
        top_k: int = KNN_DEFAULT_TOP_K,
        metric: str = "euclidean",
        algorithm: str = "kd_tree",
        n_probe_clusters: int = 1,
        include_advanced_strategies: bool = False,
        backend: str = "kdtree",
    ) -> PipelineQueryResult:
        """Run the full pipeline starting from an audio file on disk."""
        t0 = time.perf_counter()
        extraction = extract_features_from_path(file_path)
        return self._run_retrieval(
            extraction=extraction,
            mode=mode,
            top_k=top_k,
            metric=metric,
            algorithm=algorithm,
            n_probe_clusters=n_probe_clusters,
            include_advanced_strategies=include_advanced_strategies,
            backend=backend,
            t_start=t0,
        )

    def query_dataset_sample(
        self,
        row_index: int,
        mode: str = "compare",
        top_k: int = KNN_DEFAULT_TOP_K,
        metric: str = "euclidean",
        algorithm: str = "kd_tree",
        n_probe_clusters: int = 1,
        include_advanced_strategies: bool = False,
        backend: str = "kdtree",
    ) -> PipelineQueryResult:
        """Query using a sample from the 648-file dataset (uses real WAV if found, else raw 92D CSV)."""
        row = self.df_raw.iloc[int(row_index)]
        resolved = resolve_audio_path(str(row["path"]))
        if resolved is not None:
            return self.query_audio_path(
                resolved,
                mode=mode,
                top_k=top_k,
                metric=metric,
                algorithm=algorithm,
                n_probe_clusters=n_probe_clusters,
                include_advanced_strategies=include_advanced_strategies,
                backend=backend,
            )

        # Fallback if original WAV files are unavailable on the host
        t0 = time.perf_counter()
        feat_cols = self.scaler.feature_names
        vec_raw = row[feat_cols].to_numpy(dtype=np.float64)
        duration = float(row["duration"])
        num_samples = int(round(duration * 16000))
        metadata = AudioMetadata(
            filename=str(row["name"]),
            file_format="WAV (CSV vector)",
            file_size_bytes=int(row["size"]),
            original_sample_rate=int(row["frame_rate"]),
            resampled_sample_rate=16000,
            original_channels=int(row["channels"]),
            duration_sec=duration,
            num_samples=num_samples,
            num_frames_hop=max(1, int(np.ceil(num_samples / 160.0))),
            num_stft_frames=max(1, 1 + num_samples // 160),
            extraction_time_ms=0.0,
        )
        extraction = FeatureExtractionResult(
            vector_raw=vec_raw,
            feature_names=feat_cols,
            metadata=metadata,
            waveform=np.zeros(0, dtype=np.float32),
        )
        return self._run_retrieval(
            extraction=extraction,
            mode=mode,
            top_k=top_k,
            metric=metric,
            algorithm=algorithm,
            n_probe_clusters=n_probe_clusters,
            include_advanced_strategies=include_advanced_strategies,
            backend=backend,
            t_start=t0,
        )
