"""Tests for K-Means (K=13) and KNN (Full-scan, Cluster-restricted, Fallback, Multi-cluster, PCA)."""

from __future__ import annotations

import numpy as np

from web_app.config import KMEANS_MODEL_PATH, KMEANS_N_CLUSTERS, KNN_INDEX_PATH
from web_app.engine.kmeans_engine import KMeansEngine
from web_app.engine.knn_engine import KNNEngine


def test_kmeans_loaded_model_properties_and_predict() -> None:
    km = KMeansEngine.load(KMEANS_MODEL_PATH)
    assert km.n_clusters == KMEANS_N_CLUSTERS
    assert len(km.cluster_labels) == 648
    for metric_key in ("silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi", "purity"):
        assert metric_key in km.metrics

    knn = KNNEngine.load(KNN_INDEX_PATH)
    pred = km.predict(knn.X_zscore[0])
    assert 0 <= pred.cluster_id < KMEANS_N_CLUSTERS
    assert pred.distance_to_centroid >= 0.0
    assert len(pred.nearest_clusters_order) == KMEANS_N_CLUSTERS
    assert pred.nearest_clusters_order[0] == pred.cluster_id
    assert pred.cluster_size > 0


def test_knn_full_scan_and_algorithms() -> None:
    knn = KNNEngine.load(KNN_INDEX_PATH)
    q = knn.X_zscore[10]

    res_kd = knn.search_full_scan(q, top_k=5, metric="euclidean", algorithm="kd_tree")
    res_bt = knn.search_full_scan(q, top_k=5, metric="euclidean", algorithm="ball_tree")
    res_br = knn.search_full_scan(q, top_k=5, metric="euclidean", algorithm="brute")
    res_cos = knn.search_full_scan(q, top_k=5, metric="cosine", algorithm="kd_tree")

    for res in (res_kd, res_bt, res_br, res_cos):
        assert len(res.matches) == 5
        assert res.candidates_scanned == 648
        dists = [m.distance for m in res.matches]
        assert dists == sorted(dists)
        assert res.matches[0].dataset_index == 10

    assert [m.dataset_index for m in res_kd.matches] == [m.dataset_index for m in res_br.matches]


def test_knn_with_kmeans_single_multi_fallback_and_pca() -> None:
    km = KMeansEngine.load(KMEANS_MODEL_PATH)
    knn = KNNEngine.load(KNN_INDEX_PATH)
    q = knn.X_zscore[25]

    res_c1, pred = knn.search_with_kmeans(q, km, top_k=5, n_probe_clusters=1)
    assert len(res_c1.matches) == 5
    assert res_c1.candidates_scanned < 648
    assert res_c1.clusters_searched[0] == pred.cluster_id

    res_c2, _ = knn.search_with_kmeans(q, km, top_k=5, n_probe_clusters=2)
    assert len(res_c2.matches) == 5
    assert len(res_c2.clusters_searched) == 2
    assert res_c2.candidates_scanned >= res_c1.candidates_scanned

    # Force fallback by requesting more neighbors than any single cluster holds
    largest_cluster = max(len(v) for v in km.cluster_members.values())
    res_fb, _ = knn.search_with_kmeans(q, km, top_k=largest_cluster + 5, n_probe_clusters=1)
    assert res_fb.fallback_used is True
    assert len(res_fb.matches) == largest_cluster + 5
    assert len(res_fb.clusters_searched) >= 2

    # Strategy 4: PCA(92->20) + KMeans + KNN
    res_pca = knn.search_pca_kmeans(q, top_k=5)
    assert len(res_pca.matches) == 5
    assert res_pca.candidates_scanned < 648
