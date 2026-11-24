"""Truy vấn K-Means + KNN trực tiếp trong PostgreSQL bằng extension pgvector."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from web_app.config import PROJECT_ROOT, resolve_audio_path
from web_app.engine.kmeans_engine import KMeansPrediction
from web_app.engine.knn_engine import KNNSearchResult, NeighborMatch, _vote_species

load_dotenv(PROJECT_ROOT / ".env")

def _to_numpy(vec) -> np.ndarray:
    if hasattr(vec, "to_numpy"):
        return vec.to_numpy().astype(np.float64).reshape(-1)
    return np.asarray(vec, dtype=np.float64).reshape(-1)


def _to_list(vec) -> list[float]:
    return _to_numpy(vec).tolist()

class PgVectorRepository:
    """Thực thi truy vấn tìm kiếm vector 92D trực tiếp trên PostgreSQL (pgvector)."""

    def __init__(self) -> None:
        self.host = os.getenv("PG_HOST", "localhost")
        self.port = int(os.getenv("PG_PORT", "5433"))
        self.dbname = os.getenv("PG_DBNAME", "animal_audio_oop_db")
        self.user = os.getenv("PG_USER", "postgres")
        self.password = os.getenv("PG_PASSWORD", "13062005@An")

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        register_vector(conn)
        return conn

    def health_check(self) -> dict[str, Any]:
        """Kiểm tra kết nối PostgreSQL và đếm số bản ghi hiện có."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM animal_audio;")
                n_files = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM kmeans_clusters;")
                n_clusters = int(cur.fetchone()[0])
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "n_files": n_files,
            "n_clusters": n_clusters,
        }

    def predict_cluster_sql(self, query_zscore: np.ndarray) -> KMeansPrediction:
        """Tìm cụm K-Means gần nhất bằng SQL pgvector (<-> : khoảng cách Euclidean L2)."""
        q_list = _to_list(query_zscore)

        sql = """
            SELECT cluster_id,
                   centroid,
                   cluster_size,
                   species_distribution,
                   (centroid <-> %s::vector) AS dist
            FROM kmeans_clusters
            ORDER BY centroid <-> %s::vector;
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (q_list, q_list))
                rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n_clusters = len(rows)
        nearest_order = [int(r[0]) for r in rows]
        all_dists = np.zeros(n_clusters, dtype=np.float64)
        for r in rows:
            all_dists[int(r[0])] = float(r[4])

        best_id, best_centroid, best_size, best_dist_json, best_dist = rows[0]
        return KMeansPrediction(
            n_clusters=n_clusters,
            cluster_id=int(best_id),
            centroid=_to_numpy(best_centroid),
            distance_to_centroid=float(best_dist),
            all_centroid_distances=all_dists,
            nearest_clusters_order=nearest_order,
            cluster_size=int(best_size),
            species_distribution=dict(best_dist_json or {}),
            predict_time_ms=elapsed_ms,
        )

    def _rows_to_matches(self, rows: list[tuple]) -> list[NeighborMatch]:
        matches: list[NeighborMatch] = []
        for rank, r in enumerate(rows, start=1):
            db_id, name, species, fpath, cid, dur, sr, fsize, dist = r
            matches.append(
                NeighborMatch(
                    rank=rank,
                    dataset_index=int(db_id) - 1,
                    name=str(name),
                    species=str(species),
                    csv_path=str(fpath),
                    cluster_id=int(cid),
                    distance=float(dist),
                    duration=float(dur),
                    frame_rate=int(sr),
                    size=int(fsize),
                    audio_path=resolve_audio_path(str(fpath)),
                )
            )
        return matches

    def search_full_scan_sql(
        self,
        query_zscore: np.ndarray,
        top_k: int = 5,
        metric: str = "euclidean",
    ) -> KNNSearchResult:
        """Bản KHÔNG K-Means: truy vấn KNN trên toàn bộ 648 file bằng SQL pgvector."""
        q_list = _to_list(query_zscore)
        op = "<=>" if metric == "cosine" else "<->"

        sql = f"""
            SELECT id, file_name, species, file_path, cluster_id,
                   duration_sec, (tech_specs).frame_rate, (tech_specs).file_size,
                   (features_zscore {op} %s::vector) AS distance
            FROM animal_audio
            ORDER BY features_zscore {op} %s::vector
            LIMIT %s;
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (q_list, q_list, int(top_k)))
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) FROM animal_audio;")
                total_candidates = int(cur.fetchone()[0])
        search_ms = (time.perf_counter() - t0) * 1000.0

        matches = self._rows_to_matches(rows)
        maj_sp, wt_sp, votes = _vote_species(matches)

        return KNNSearchResult(
            mode_name="PostgreSQL pgvector — Không K-means (Full-scan)",
            algorithm=f"pgvector SQL ({op})",
            metric=metric,
            n_neighbors=len(matches),
            candidates_scanned=total_candidates,
            clusters_searched=list(range(13)),
            fallback_used=False,
            kmeans_predict_time_ms=0.0,
            search_time_ms=search_ms,
            total_retrieval_time_ms=search_ms,
            matches=matches,
            predicted_species_majority=maj_sp,
            predicted_species_weighted=wt_sp,
            species_votes_weighted=votes,
        )

    def search_with_kmeans_sql(
        self,
        query_zscore: np.ndarray,
        top_k: int = 5,
        n_probe_clusters: int = 1,
        metric: str = "euclidean",
        prediction: KMeansPrediction | None = None,
    ) -> tuple[KNNSearchResult, KMeansPrediction]:
        """Bản CÓ K-Means: lọc WHERE cluster_id = ANY(%s) rồi tìm KNN bằng SQL pgvector."""
        q_list = _to_list(query_zscore)
        op = "<=>" if metric == "cosine" else "<->"

        pred = prediction if prediction is not None else self.predict_cluster_sql(query_zscore)
        n_probe = max(1, min(int(n_probe_clusters), pred.n_clusters))
        selected_clusters = list(pred.nearest_clusters_order[:n_probe])
        fallback_used = False

        sql_count = "SELECT COUNT(*) FROM animal_audio WHERE cluster_id = ANY(%s);"
        sql_knn = f"""
            SELECT id, file_name, species, file_path, cluster_id,
                   duration_sec, (tech_specs).frame_rate, (tech_specs).file_size,
                   (features_zscore {op} %s::vector) AS distance
            FROM animal_audio
            WHERE cluster_id = ANY(%s)
            ORDER BY features_zscore {op} %s::vector
            LIMIT %s;
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_count, (selected_clusters,))
                candidates_scanned = int(cur.fetchone()[0])

                # Fallback: nếu cụm gốc có ít hơn top_k file thì mở rộng sang cụm kế cận
                next_pos = n_probe
                while candidates_scanned < int(top_k) and next_pos < len(pred.nearest_clusters_order):
                    fallback_used = True
                    selected_clusters.append(pred.nearest_clusters_order[next_pos])
                    next_pos += 1
                    cur.execute(sql_count, (selected_clusters,))
                    candidates_scanned = int(cur.fetchone()[0])

                cur.execute(sql_knn, (q_list, selected_clusters, q_list, int(top_k)))
                rows = cur.fetchall()
        search_ms = (time.perf_counter() - t0) * 1000.0

        matches = self._rows_to_matches(rows)
        maj_sp, wt_sp, votes = _vote_species(matches)

        result = KNNSearchResult(
            mode_name=f"PostgreSQL pgvector — Có K-means (K={pred.n_clusters}, Top-{len(selected_clusters)} cụm)",
            algorithm=f"pgvector SQL WHERE cluster_id ({op})",
            metric=metric,
            n_neighbors=len(matches),
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


if __name__ == "__main__":
    # Kiểm thử nhanh module pgvector độc lập
    repo = PgVectorRepository()
    print("Health check:", repo.health_check())

    # Lấy thử vector đầu tiên trong bảng để test truy vấn
    with repo._connect() as conn:
        q_vec = conn.execute("SELECT features_zscore FROM animal_audio LIMIT 1;").fetchone()[0]

    res_full = repo.search_full_scan_sql(q_vec, top_k=5)
    res_km, pred = repo.search_with_kmeans_sql(q_vec, top_k=5)
    print(f"Full-scan SQL ({res_full.search_time_ms:.2f} ms): {[m.name for m in res_full.matches]}")
    print(f"K-Means SQL Cụm #{pred.cluster_id} ({res_km.search_time_ms:.2f} ms, {res_km.candidates_scanned} file): {[m.name for m in res_km.matches]}")