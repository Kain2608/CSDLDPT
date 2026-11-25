"""Truy vấn K-Means + KNN trực tiếp trong PostgreSQL qua Stored Methods (OOP + pgvector)."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from web_app.config import PROJECT_ROOT, resolve_audio_path
from web_app.engine.kmeans_engine import KMeansPrediction
from web_app.engine.knn_engine import KNNSearchResult, NeighborMatch, _vote_species

load_dotenv(PROJECT_ROOT / ".env")


def _to_numpy(vec: Any) -> np.ndarray:
    """Chuyển đổi an toàn pgvector.Vector / list / ndarray về vector 1D float64."""
    if hasattr(vec, "to_numpy"):
        return vec.to_numpy().astype(np.float64).reshape(-1)
    return np.asarray(vec, dtype=np.float64).reshape(-1)


def _to_list(vec: Any) -> list[float]:
    return _to_numpy(vec).tolist()


class PgVectorRepository:
    """Giao tiếp với CSDL OOP PostgreSQL (animal_audio_oop_db) bằng Stored Methods + pgvector."""

    def __init__(self) -> None:
        self.host = os.getenv("PG_HOST", "localhost")
        self.port = int(os.getenv("PG_PORT", "5433"))
        self.dbname = os.getenv("PG_DBNAME", "animal_audio_oop_db")
        self.user = os.getenv("PG_USER", "postgres")
        self.password = os.getenv("PG_PASSWORD", "")

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
        """Kiểm tra kết nối PostgreSQL, tính kế thừa lớp multimedia_entity -> animal_audio và số cụm."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM multimedia_entity;")
                n_parent = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM animal_audio;")
                n_files = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM kmeans_clusters;")
                n_clusters = int(cur.fetchone()[0])
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "multimedia_entity_count": n_parent,
            "n_files": n_files,
            "n_clusters": n_clusters,
        }

    def predict_cluster_sql(self, query_zscore: np.ndarray) -> KMeansPrediction:
        """Gọi Stored Method `predict_nearest_clusters(vector)` trong PostgreSQL."""
        q_list = _to_list(query_zscore)

        sql = """
            SELECT cluster_id, centroid, cluster_size, species_distribution, distance
            FROM predict_nearest_clusters(%s::vector);
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (q_list,))
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
        for r in rows:
            rank_no, oid, name, species, fpath, cid, dur, sr, fsize, dist, _sim = r
            matches.append(
                NeighborMatch(
                    rank=int(rank_no),
                    dataset_index=int(oid) - 1,
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
        feature_set: str = "C",
    ) -> KNNSearchResult:
        """Bản KHÔNG K-Means: gọi phương thức `search_similar_animals(..., NULL, top_k)` trên toàn bộ 648 file."""
        q_list = _to_list(query_zscore)
        op = "<=>" if metric == "cosine" else "<->"

        sql = """
            SELECT rank_no, oid, file_name, species, file_path, cluster_id,
                   duration_sec, frame_rate, size_bytes, distance, similarity_score
            FROM search_similar_animals(%s::vector, %s, %s, NULL, %s);
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (q_list, feature_set.upper(), metric.lower(), int(top_k)))
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) FROM animal_audio;")
                total_candidates = int(cur.fetchone()[0])
        search_ms = (time.perf_counter() - t0) * 1000.0

        matches = self._rows_to_matches(rows)
        maj_sp, wt_sp, votes = _vote_species(matches)

        return KNNSearchResult(
            mode_name=f"PostgreSQL pgvector — Không K-means (Full-scan, Bộ {feature_set.upper()})",
            algorithm=f"pgvector Stored Method ({op})",
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
        feature_set: str = "C",
        prediction: KMeansPrediction | None = None,
    ) -> tuple[KNNSearchResult, KMeansPrediction]:
        """Bản CÓ K-Means: gọi `search_similar_animals(..., target_clusters, top_k)` có hỗ trợ Fallback."""
        q_list = _to_list(query_zscore)
        op = "<=>" if metric == "cosine" else "<->"

        pred = prediction if prediction is not None else self.predict_cluster_sql(query_zscore)
        n_probe = max(1, min(int(n_probe_clusters), pred.n_clusters))
        selected_clusters = list(pred.nearest_clusters_order[:n_probe])
        fallback_used = False

        sql_count = "SELECT COUNT(*) FROM animal_audio WHERE cluster_id = ANY(%s);"
        sql_knn = """
            SELECT rank_no, oid, file_name, species, file_path, cluster_id,
                   duration_sec, frame_rate, size_bytes, distance, similarity_score
            FROM search_similar_animals(%s::vector, %s, %s, %s::int[], %s);
        """

        t0 = time.perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_count, (selected_clusters,))
                candidates_scanned = int(cur.fetchone()[0])

                # Fallback: nếu cụm gốc có ít hơn top_k file thì tự mở rộng sang cụm kế cận
                next_pos = n_probe
                while candidates_scanned < int(top_k) and next_pos < len(pred.nearest_clusters_order):
                    fallback_used = True
                    selected_clusters.append(pred.nearest_clusters_order[next_pos])
                    next_pos += 1
                    cur.execute(sql_count, (selected_clusters,))
                    candidates_scanned = int(cur.fetchone()[0])

                cur.execute(
                    sql_knn,
                    (q_list, feature_set.upper(), metric.lower(), selected_clusters, int(top_k)),
                )
                rows = cur.fetchall()
        search_ms = (time.perf_counter() - t0) * 1000.0

        matches = self._rows_to_matches(rows)
        maj_sp, wt_sp, votes = _vote_species(matches)

        result = KNNSearchResult(
            mode_name=f"PostgreSQL pgvector — Có K-means (K={pred.n_clusters}, Top-{len(selected_clusters)} cụm, Bộ {feature_set.upper()})",
            algorithm=f"pgvector Stored Method + cluster_id ({op})",
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
    repo = PgVectorRepository()
    print("Health check:", repo.health_check())

    with repo._connect() as conn:
        q_vec = conn.execute("SELECT features_zscore FROM animal_audio WHERE file_name = 'Bear_1.wav';").fetchone()[0]

    for fset in ("A", "B", "C"):
        res_full = repo.search_full_scan_sql(q_vec, top_k=5, feature_set=fset)
        print(f"Bộ {fset} Full-scan ({res_full.search_time_ms:.2f} ms): {[m.name for m in res_full.matches]}")

    res_km, pred = repo.search_with_kmeans_sql(q_vec, top_k=5, feature_set="C")
    print(
        f"Bộ C K-Means Cụm #{pred.cluster_id} ({res_km.search_time_ms:.2f} ms, {res_km.candidates_scanned} file): "
        f"{[m.name for m in res_km.matches]}"
    )
