"""Offline benchmark & static report chart generation (Sections 3, 4, 6, 8.4)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models import train_all
from web_app.config import (
    BENCHMARK_N_QUERIES,
    BENCHMARK_RANDOM_STATE,
    ELBOW_SILHOUETTE_PNG,
    KMEANS_INIT,
    KMEANS_MODEL_PATH,
    KMEANS_N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    KNN_INDEX_PATH,
    K_SWEEP_RANGE,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PCA_N_COMPONENTS,
    PCA_SCATTER_PNG,
    SCALER_PARAMS_JSON,
    SPECIES_CLUSTER_HEATMAP_PNG,
    SPEED_BENCHMARK_KS,
    STATIC_DIR,
    ZSCORE_FEATURES_CSV,
)
from web_app.engine.kmeans_engine import KMeansEngine, sweep_kmeans_k
from web_app.engine.scaler import ZScoreScaler


def benchmark_knn_algorithms(
    X: np.ndarray,
    query_indices: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compare KDTree, BallTree, and Brute Force query latency (Section 3.4)."""
    rows: list[dict[str, Any]] = []
    for algo in ("kd_tree", "ball_tree", "brute"):
        t_fit0 = time.perf_counter()
        nn = NearestNeighbors(algorithm=algo, metric="euclidean")
        nn.fit(X)
        fit_ms = (time.perf_counter() - t_fit0) * 1000.0

        # Warm-up 5 queries
        for idx in query_indices[:5]:
            nn.kneighbors(X[idx : idx + 1], n_neighbors=top_k + 1)

        t0 = time.perf_counter()
        for idx in query_indices:
            nn.kneighbors(X[idx : idx + 1], n_neighbors=top_k + 1)
        total_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = total_ms / len(query_indices)

        rows.append({
            "algorithm": algo,
            "fit_time_ms": round(fit_ms, 4),
            "avg_query_ms": round(avg_ms, 4),
            "n_queries": int(len(query_indices)),
        })

    kdtree_ms = rows[0]["avg_query_ms"] or 1e-6
    for r in rows:
        r["relative_to_kdtree"] = round(r["avg_query_ms"] / kdtree_ms, 2)
    return rows


def _query_subset_top_k_excluding_self(
    X_sub: np.ndarray,
    global_indices: np.ndarray,
    q: np.ndarray,
    self_idx: int,
    top_k: int,
) -> np.ndarray:
    """Return global indices of Top-K Euclidean neighbors in `X_sub`, excluding `self_idx`."""
    dists = np.linalg.norm(X_sub - q, axis=1)
    order = np.argsort(dists)
    result: list[int] = []
    for pos in order:
        g_idx = int(global_indices[pos])
        if g_idx != self_idx:
            result.append(g_idx)
            if len(result) == top_k:
                break
    return np.asarray(result, dtype=np.int64)


def benchmark_full_vs_kmeans_speed(
    X: np.ndarray,
    species: np.ndarray,
    query_indices: np.ndarray,
    k_list: list[int] = SPEED_BENCHMARK_KS,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Measure average query latency, same-species Recall@5, and overlap@5 across Full-scan & K values (Section 3.3)."""
    n_samples = len(X)
    all_indices = np.arange(n_samples)

    # Baseline: Full-scan direct vectorized Euclidean (matching Section 3.3 full-scan speed benchmark)
    # Warm-up
    for idx in query_indices[:5]:
        _query_subset_top_k_excluding_self(X, all_indices, X[idx], int(idx), top_k)

    full_neighbors: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for idx in query_indices:
        full_neighbors[int(idx)] = _query_subset_top_k_excluding_self(X, all_indices, X[idx], int(idx), top_k)
    full_avg_ms = ((time.perf_counter() - t0) * 1000.0) / len(query_indices)

    full_species_hits = [
        float(np.mean(species[full_neighbors[int(idx)]] == species[int(idx)]))
        for idx in query_indices
    ]
    full_recall = float(np.mean(full_species_hits))

    rows: list[dict[str, Any]] = [
        {
            "method": "Full-scan (không K-means)",
            "k": None,
            "avg_candidates": float(n_samples),
            "avg_query_ms": round(full_avg_ms, 4),
            "recall_at_5": round(full_recall, 4),
            "overlap_with_full_at_5": 1.0,
            "speedup_vs_full": 1.0,
        }
    ]

    for k in k_list:
        km = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
        )
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        cluster_members = {c: np.where(labels == c)[0] for c in range(k)}
        cluster_arrays = {c: X[idxs] for c, idxs in cluster_members.items()}

        # Warm-up
        for idx in query_indices[:5]:
            c_order = np.argsort(np.linalg.norm(centers - X[idx], axis=1))
            c0 = int(c_order[0])
            _query_subset_top_k_excluding_self(cluster_arrays[c0], cluster_members[c0], X[idx], int(idx), top_k)

        cand_counts: list[int] = []
        species_hits: list[float] = []
        overlaps: list[float] = []

        t_k0 = time.perf_counter()
        for idx in query_indices:
            q = X[idx]
            c_dists = np.linalg.norm(centers - q, axis=1)
            c0 = int(np.argmin(c_dists))
            mem_idx = cluster_members[c0]
            sub_X = cluster_arrays[c0]

            # Fallback if cluster has <= top_k elements (excluding self)
            if len(mem_idx) <= top_k:
                order_c = np.argsort(c_dists)
                parts = [cluster_members[int(c)] for c in order_c[:2]]
                pos = 2
                merged = np.concatenate(parts)
                while len(merged) <= top_k and pos < len(order_c):
                    merged = np.concatenate([merged, cluster_members[int(order_c[pos])]])
                    pos += 1
                mem_idx = merged
                sub_X = X[mem_idx]

            nbrs = _query_subset_top_k_excluding_self(sub_X, mem_idx, q, int(idx), top_k)
            cand_counts.append(len(mem_idx))
            species_hits.append(float(np.mean(species[nbrs] == species[int(idx)])))
            overlaps.append(len(set(nbrs.tolist()) & set(full_neighbors[int(idx)].tolist())) / float(top_k))

        avg_ms = ((time.perf_counter() - t_k0) * 1000.0) / len(query_indices)

        rows.append({
            "method": f"K-means K={k} + KNN trong cụm",
            "k": int(k),
            "avg_candidates": round(float(np.mean(cand_counts)), 1),
            "avg_query_ms": round(avg_ms, 4),
            "recall_at_5": round(float(np.mean(species_hits)), 4),
            "overlap_with_full_at_5": round(float(np.mean(overlaps)), 4),
            "speedup_vs_full": round(full_avg_ms / max(avg_ms, 1e-6), 2),
        })

    return rows


def benchmark_four_strategies(
    X: np.ndarray,
    species: np.ndarray,
    query_indices: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Benchmark the 4 retrieval strategies from Section 8.4."""
    n_samples = len(X)
    all_indices = np.arange(n_samples)

    # Fit KMeans(K=13) in 92D
    km13 = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    labels13 = km13.fit_predict(X)
    centers13 = km13.cluster_centers_
    members13 = {c: np.where(labels13 == c)[0] for c in range(KMEANS_N_CLUSTERS)}

    # Fit PCA(92 -> 20) + StandardScaler + KMeans(K=13)
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=KMEANS_RANDOM_STATE)
    X_pca = StandardScaler().fit_transform(pca.fit_transform(X))
    km_pca = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    labels_pca = km_pca.fit_predict(X_pca)
    centers_pca = km_pca.cluster_centers_
    members_pca = {c: np.where(labels_pca == c)[0] for c in range(KMEANS_N_CLUSTERS)}

    def _run_eval(name: str, fn) -> dict[str, Any]:
        for idx in query_indices[:5]:
            fn(int(idx))
        cands, hits = [], []
        t0 = time.perf_counter()
        for idx in query_indices:
            nbrs, n_cand = fn(int(idx))
            cands.append(n_cand)
            hits.append(float(np.mean(species[nbrs] == species[int(idx)])))
        avg_ms = ((time.perf_counter() - t0) * 1000.0) / len(query_indices)
        return {
            "strategy": name,
            "avg_candidates": round(float(np.mean(cands)), 1),
            "avg_query_ms": round(avg_ms, 4),
            "recall_at_5": round(float(np.mean(hits)), 4),
        }

    def s1(idx: int):
        return _query_subset_top_k_excluding_self(X, all_indices, X[idx], idx, top_k), n_samples

    def s2(idx: int):
        q = X[idx]
        c_order = np.argsort(np.linalg.norm(centers13 - q, axis=1))
        m = members13[int(c_order[0])]
        if len(m) <= top_k:
            m = np.concatenate([m, members13[int(c_order[1])]])
        return _query_subset_top_k_excluding_self(X[m], m, q, idx, top_k), len(m)

    def s3(idx: int):
        q = X[idx]
        c_order = np.argsort(np.linalg.norm(centers13 - q, axis=1))
        m = np.concatenate([members13[int(c_order[0])], members13[int(c_order[1])]])
        return _query_subset_top_k_excluding_self(X[m], m, q, idx, top_k), len(m)

    def s4(idx: int):
        q = X_pca[idx]
        c_order = np.argsort(np.linalg.norm(centers_pca - q, axis=1))
        m = members_pca[int(c_order[0])]
        if len(m) <= top_k:
            m = np.concatenate([m, members_pca[int(c_order[1])]])
        return _query_subset_top_k_excluding_self(X_pca[m], m, q, idx, top_k), len(m)

    results = [
        _run_eval("Chiến lược 1: Full-scan (không K-means)", s1),
        _run_eval("Chiến lược 2: K-means K=13 → KNN Top-1 cụm", s2),
        _run_eval("Chiến lược 3: K-means K=13 → KNN Top-2 cụm", s3),
        _run_eval("Chiến lược 4: PCA(92→20) + K-means K=13 + KNN", s4),
    ]
    base_ms = results[0]["avg_query_ms"] or 1e-6
    for r in results:
        r["speedup_vs_full"] = round(base_ms / max(r["avg_query_ms"], 1e-6), 2)
    return results


def generate_static_charts(
    k_sweep_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
) -> None:
    """Generate publication-quality static PNG charts in `web_app/static/`."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Elbow + Silhouette + Davies-Bouldin chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    ax0 = axes[0]
    ax0.plot(k_sweep_df["k"], k_sweep_df["inertia"], marker="o", linewidth=2, color="#2563EB", label="Inertia (SSE)")
    k13_row = k_sweep_df[k_sweep_df["k"] == KMEANS_N_CLUSTERS].iloc[0]
    ax0.axvline(KMEANS_N_CLUSTERS, color="#DC2626", linestyle="--", linewidth=1.6, label=f"K={KMEANS_N_CLUSTERS} (Chọn)")
    ax0.scatter([KMEANS_N_CLUSTERS], [k13_row["inertia"]], color="#DC2626", s=80, zorder=5)
    ax0.set_title("Phương pháp Elbow (Inertia theo số cụm K)", fontweight="bold")
    ax0.set_xlabel("Số cụm K")
    ax0.set_ylabel("Inertia (SSE)")
    ax0.set_xticks(k_sweep_df["k"][::2])
    ax0.legend()

    ax1 = axes[1]
    l1 = ax1.plot(
        k_sweep_df["k"],
        k_sweep_df["silhouette"],
        marker="o",
        linewidth=2,
        color="#059669",
        label="Silhouette ↑",
    )
    ax1.axvline(KMEANS_N_CLUSTERS, color="#DC2626", linestyle="--", linewidth=1.6, label=f"K={KMEANS_N_CLUSTERS} (Chọn)")
    ax1.set_title("Silhouette & Davies-Bouldin theo số cụm K", fontweight="bold")
    ax1.set_xlabel("Số cụm K")
    ax1.set_ylabel("Silhouette Score ↑", color="#059669")
    ax1.set_xticks(k_sweep_df["k"][::2])

    ax1_r = ax1.twinx()
    l2 = ax1_r.plot(
        k_sweep_df["k"],
        k_sweep_df["davies_bouldin"],
        marker="s",
        linewidth=1.8,
        linestyle="-.",
        color="#D97706",
        label="Davies-Bouldin ↓",
    )
    ax1_r.set_ylabel("Davies-Bouldin Index ↓", color="#D97706")
    ax1_r.grid(False)
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(ELBOW_SILHOUETTE_PNG, bbox_inches="tight")
    plt.close(fig)

    # 2. PCA 2D Scatter (Clusters vs True Species)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    cmap = plt.get_cmap("tab20")

    ax_c = axes[0]
    for c_id in sorted(proj_df["cluster_id"].unique()):
        sub = proj_df[proj_df["cluster_id"] == c_id]
        ax_c.scatter(sub["pca_x"], sub["pca_y"], s=24, alpha=0.78, color=cmap(c_id % 20), label=f"C{c_id}")
    ax_c.scatter(
        centroids_df["pca_x"],
        centroids_df["pca_y"],
        s=150,
        marker="X",
        color="#111827",
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
        label="Centroids",
    )
    ax_c.set_title("PCA 2D — Phân bố theo 13 cụm K-Means", fontweight="bold")
    ax_c.set_xlabel("PC1")
    ax_c.set_ylabel("PC2")
    ax_c.legend(ncol=3, fontsize=7.5, loc="best", framealpha=0.85)

    ax_s = axes[1]
    species_sorted = sorted(proj_df["species"].unique())
    for idx_sp, sp in enumerate(species_sorted):
        sub = proj_df[proj_df["species"] == sp]
        ax_s.scatter(sub["pca_x"], sub["pca_y"], s=24, alpha=0.78, color=cmap(idx_sp % 20), label=sp)
    ax_s.set_title("PCA 2D — Phân bố theo 13 loài thực tế", fontweight="bold")
    ax_s.set_xlabel("PC1")
    ax_s.set_ylabel("PC2")
    ax_s.legend(ncol=2, fontsize=7.5, loc="best", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(PCA_SCATTER_PNG, bbox_inches="tight")
    plt.close(fig)

    # 3. Species x Cluster Contingency Heatmap
    ct = pd.crosstab(proj_df["species"], proj_df["cluster_id"])
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    im = ax.imshow(ct.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(ct.columns)))
    ax.set_xticklabels([f"Cụm {c}" for c in ct.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ct.index)))
    ax.set_yticklabels(ct.index)
    ax.set_xlabel("Cụm K-Means (K=13)")
    ax.set_ylabel("Loài động vật (13 loài)")
    ax.set_title("Ma trận phân bố Loài × Cụm K-Means (648 tệp)", fontweight="bold")
    ax.grid(False)

    mat = ct.to_numpy()
    thresh = mat.max() * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val >= thresh else "#111827",
                    fontsize=8.5,
                    fontweight="bold" if val >= thresh else "normal",
                )

    fig.colorbar(im, ax=ax, label="Số tệp WAV")
    fig.tight_layout()
    fig.savefig(SPECIES_CLUSTER_HEATMAP_PNG, bbox_inches="tight")
    plt.close(fig)


def run_all_benchmarks() -> dict[str, Any]:
    """Execute all offline benchmarks, update model_metadata.json, and export static charts."""
    if not KMEANS_MODEL_PATH.is_file() or not KNN_INDEX_PATH.is_file():
        train_all()

    df_zscore = pd.read_csv(ZSCORE_FEATURES_CSV)
    scaler = ZScoreScaler.from_json(SCALER_PARAMS_JSON)
    X = df_zscore[scaler.feature_names].to_numpy(dtype=np.float64)
    species = df_zscore["species"].astype(str).to_numpy()

    rng = np.random.default_rng(BENCHMARK_RANDOM_STATE)
    n_queries = min(BENCHMARK_N_QUERIES, len(X))
    query_indices = rng.choice(len(X), size=n_queries, replace=False)

    print(f"[run_benchmark] 1/4 Sweeping K in {K_SWEEP_RANGE[0]}..{K_SWEEP_RANGE[-1]}...")
    k_sweep = sweep_kmeans_k(X, species, k_values=K_SWEEP_RANGE)

    print(f"[run_benchmark] 2/4 Benchmarking KNN algorithms (KDTree vs BallTree vs Brute, {n_queries} queries)...")
    algo_bench = benchmark_knn_algorithms(X, query_indices, top_k=5)

    print(f"[run_benchmark] 3/4 Benchmarking Full-scan vs K-Means speed & Recall@5 ({n_queries} queries)...")
    speed_bench = benchmark_full_vs_kmeans_speed(X, species, query_indices, k_list=SPEED_BENCHMARK_KS, top_k=5)

    print(f"[run_benchmark] 4/4 Benchmarking 4 retrieval strategies (Section 8.4, {n_queries} queries)...")
    strategy_bench = benchmark_four_strategies(X, species, query_indices, top_k=5)

    # Generate static PNGs
    proj_df = pd.read_csv(MODELS_DIR / "projections_2d.csv")
    centroids_df = pd.read_csv(MODELS_DIR / "centroids_pca_2d.csv")
    generate_static_charts(pd.DataFrame(k_sweep), proj_df, centroids_df)
    print(f"[run_benchmark] Saved static charts to {STATIC_DIR}")

    # Persist into model_metadata.json
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["benchmarks"] = {
        "n_queries": int(n_queries),
        "random_state": BENCHMARK_RANDOM_STATE,
        "k_sweep": k_sweep,
        "knn_algorithms": algo_bench,
        "full_vs_kmeans_speed": speed_bench,
        "four_strategies": strategy_bench,
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("[run_benchmark] Completed successfully!")
    return metadata["benchmarks"]


if __name__ == "__main__":
    run_all_benchmarks()
