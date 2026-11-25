"""Tests verifying the unified OOP + pgvector + K-Means PostgreSQL schema and stored methods."""

from __future__ import annotations

import pytest

from web_app.engine.pgvector_repository import PgVectorRepository
from web_app.engine.pipeline import AnimalSoundPipeline


def _get_live_repo() -> PgVectorRepository:
    repo = PgVectorRepository()
    try:
        repo.health_check()
    except Exception as exc:
        pytest.skip(f"PostgreSQL pgvector database not reachable: {exc}")
    return repo


def test_oop_inheritance_udts_and_row_counts() -> None:
    repo = _get_live_repo()
    health = repo.health_check()
    assert health["multimedia_entity_count"] == 648
    assert health["n_files"] == 648
    assert health["n_clusters"] == 13

    with repo._connect() as conn:
        with conn.cursor() as cur:
            # Verify INHERITS(multimedia_entity)
            cur.execute(
                """
                SELECT c.relname, p.relname
                FROM pg_inherits i
                JOIN pg_class c ON c.oid = i.inhrelid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE c.relname = 'animal_audio';
                """
            )
            assert cur.fetchall() == [("animal_audio", "multimedia_entity")]

            # Verify both UDTs exist and are accessible via dot syntax
            cur.execute(
                """
                SELECT oid, file_name,
                       (tech_specs).frame_rate, (tech_specs).size_bytes,
                       (proc_config).target_sample_rate, (proc_config).n_mfcc
                FROM animal_audio
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            assert row is not None
            assert row[4] == 16000
            assert row[5] == 13


@pytest.mark.parametrize("feature_set", ["A", "B", "C"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_stored_method_search_similar_animals_all_feature_sets(feature_set: str, metric: str) -> None:
    repo = _get_live_repo()
    with repo._connect() as conn:
        q_vec = conn.execute("SELECT features_zscore FROM animal_audio WHERE file_name = 'Bear_1.wav';").fetchone()[0]

    res_full = repo.search_full_scan_sql(q_vec, top_k=5, metric=metric, feature_set=feature_set)
    assert len(res_full.matches) == 5
    assert res_full.candidates_scanned == 648
    assert res_full.matches[0].name == "Bear_1.wav"
    assert res_full.matches[0].distance < 1e-4

    res_km, pred = repo.search_with_kmeans_sql(q_vec, top_k=5, metric=metric, feature_set=feature_set)
    assert len(res_km.matches) == 5
    assert res_km.candidates_scanned < 648
    assert res_km.matches[0].name == "Bear_1.wav"
    assert 0 <= pred.cluster_id < 13


def test_pgvector_stored_method_matches_python_kdtree_exactly() -> None:
    _get_live_repo()
    pipeline = AnimalSoundPipeline.load()

    res_pg = pipeline.query_dataset_sample(0, mode="compare", backend="pgvector")
    res_kd = pipeline.query_dataset_sample(0, mode="compare", backend="kdtree")

    assert [m.name for m in res_pg.result_without_kmeans.matches] == [
        m.name for m in res_kd.result_without_kmeans.matches
    ]
    assert [m.name for m in res_pg.result_with_kmeans.matches] == [
        m.name for m in res_kd.result_with_kmeans.matches
    ]
