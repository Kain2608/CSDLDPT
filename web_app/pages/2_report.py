"""Page 2 — Technical Report & Benchmarks (Sections 3, 4, 5.5, 8.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark import run_all_benchmarks
from web_app.config import (
    ELBOW_SILHOUETTE_PNG,
    KMEANS_N_CLUSTERS,
    PCA_SCATTER_PNG,
    SPECIES_CLUSTER_HEATMAP_PNG,
)
from web_app.ui_components import (
    format_ms,
    get_pipeline,
    inject_custom_css,
    load_model_metadata,
)

st.set_page_config(
    page_title="Báo cáo Kỹ thuật & Benchmark",
    page_icon="📊",
    layout="wide",
)
inject_custom_css()

st.title("📊 Báo Cáo Kỹ Thuật: K-Means + KNN Benchmark")
st.markdown("Tổng hợp đầy đủ chỉ số đánh giá phân cụm K-Means (K = 2..20), so sánh thuật toán KNN (KDTree / Ball Tree / Brute Force), và benchmark tốc độ – Recall@5 thực đo trên 200 truy vấn.")

pipeline = get_pipeline()
metadata = load_model_metadata()

with st.sidebar:
    st.header("Cập nhật Benchmark")
    st.caption("Chạy lại toàn bộ benchmark trên CPU cục bộ hiện tại và tạo lại ảnh tĩnh trong `web_app/static/`.")
    if st.button("🔄 Chạy lại Benchmark thực đo", type="primary"):
        with st.spinner("Đang chạy K-sweep (2..20) và benchmark 200 truy vấn..."):
            run_all_benchmarks()
        st.cache_data.clear()
        st.success("Đã cập nhật kết quả benchmark và biểu đồ báo cáo!")
        st.rerun()

benchmarks = metadata.get("benchmarks", {})
km_metrics = metadata.get("kmeans_metrics", pipeline.kmeans_engine.metrics)
init_timings = metadata.get("init_timings_ms", {})

# ============================================================================
# SECTION 1: K=13 SUMMARY & ONE-TIME INIT TIMINGS
# ============================================================================
st.subheader("1. Chỉ số phân cụm K-Means (K = 13) & Thời gian khởi tạo")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Silhouette ↑", f"{km_metrics.get('silhouette', 0.0):.4f}")
c2.metric("Davies-Bouldin ↓", f"{km_metrics.get('davies_bouldin', 0.0):.4f}")
c3.metric("Calinski-Harabasz ↑", f"{km_metrics.get('calinski_harabasz', 0.0):.2f}")
c4.metric("ARI ↑", f"{km_metrics.get('ari', 0.0):.4f}")
c5.metric("NMI ↑", f"{km_metrics.get('nmi', 0.0):.4f}")
c6.metric("Purity ↑", f"{km_metrics.get('purity', 0.0):.4f}")

if init_timings:
    init_df = pd.DataFrame([
        {"Bước khởi tạo (one-time)": "Đọc CSV + load tham số Z-score (μ, σ)", "Thời gian (ms)": format_ms(init_timings.get("load_csv_and_scaler_ms", 0.0))},
        {"Bước khởi tạo (one-time)": "Huấn luyện K-Means (K=13, k-means++, n_init=10)", "Thời gian (ms)": format_ms(init_timings.get("kmeans_fit_ms", 0.0))},
        {"Bước khởi tạo (one-time)": "Xây dựng cây KDTree toàn cục (648 × 92)", "Thời gian (ms)": format_ms(init_timings.get("kdtree_fit_ms", 0.0))},
    ])
    st.dataframe(init_df, width="stretch", hide_index=True)

st.divider()

# ============================================================================
# SECTION 2: K-SWEEP (ELBOW + SILHOUETTE + DAVIES-BOULDIN)
# ============================================================================
st.subheader("2. Xác định số cụm K tối ưu (K = 2..20)")

k_sweep = benchmarks.get("k_sweep", [])
if k_sweep:
    df_k = pd.DataFrame(k_sweep)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        elbow_line = (
            alt.Chart(df_k)
            .mark_line(point=True, color="#2563EB", strokeWidth=2.2)
            .encode(
                x=alt.X("k:Q", title="Số cụm K", scale=alt.Scale(domain=[2, 20])),
                y=alt.Y("inertia:Q", title="Inertia (SSE)"),
                tooltip=["k:Q", alt.Tooltip("inertia:Q", format=".1f"), alt.Tooltip("silhouette:Q", format=".4f")],
            )
        )
        k13_rule = alt.Chart(pd.DataFrame({"k": [KMEANS_N_CLUSTERS]})).mark_rule(
            color="#DC2626", strokeDash=[5, 4], strokeWidth=2
        ).encode(x="k:Q")
        st.altair_chart(
            (elbow_line + k13_rule).properties(height=300, title="Biểu đồ Elbow — Inertia theo K"),
            width="stretch",
        )

    with chart_col2:
        sil_line = (
            alt.Chart(df_k)
            .mark_line(point=True, color="#059669", strokeWidth=2.2)
            .encode(
                x=alt.X("k:Q", title="Số cụm K", scale=alt.Scale(domain=[2, 20])),
                y=alt.Y("silhouette:Q", title="Silhouette Score ↑"),
                tooltip=[
                    "k:Q",
                    alt.Tooltip("silhouette:Q", format=".4f"),
                    alt.Tooltip("davies_bouldin:Q", format=".4f"),
                    alt.Tooltip("ari:Q", format=".4f"),
                    alt.Tooltip("nmi:Q", format=".4f"),
                    alt.Tooltip("purity:Q", format=".4f"),
                ],
            )
        )
        k13_rule2 = alt.Chart(pd.DataFrame({"k": [KMEANS_N_CLUSTERS]})).mark_rule(
            color="#DC2626", strokeDash=[5, 4], strokeWidth=2
        ).encode(x="k:Q")
        st.altair_chart(
            (sil_line + k13_rule2).properties(height=300, title="Silhouette Score theo K (đường đỏ: K=13)"),
            width="stretch",
        )

    display_k = df_k.rename(columns={
        "k": "K",
        "silhouette": "Silhouette ↑",
        "ari": "ARI ↑",
        "nmi": "NMI ↑",
        "davies_bouldin": "Davies-Bouldin ↓",
        "calinski_harabasz": "Calinski-Harabasz ↑",
        "purity": "Purity ↑",
        "inertia": "Inertia (SSE)",
        "fit_time_ms": "Fit time (ms)",
    })
    st.dataframe(display_k, width="stretch", hide_index=True, height=320)

    st.info(
        "**Lý do chọn K = 13 (Mục 4.2):** Mặc dù K = 2 có Silhouette cao nhất (~0.24), K = 2 chỉ tách tập dữ liệu thành 2 nhóm lớn "
        "(ARI ≈ 0.03, Purity ≈ 0.14 — vô nghĩa đối với 13 loài). Tại **K = 13**, chỉ số **Davies-Bouldin đạt mức thấp tốt (~1.998)**, "
        "tương ứng trực tiếp 13 cụm ↔ 13 loài, Purity tăng hơn 2× (~0.306), đồng thời tốc độ tìm kiếm KNN trong cụm tăng đáng kể mà Recall@5 chỉ giảm rất ít."
    )
else:
    st.warning("Chưa có dữ liệu K-sweep. Hãy nhấn nút **Chạy lại Benchmark thực đo** ở thanh bên trái.")

st.divider()

# ============================================================================
# SECTION 3: SPEED & RECALL@5 BENCHMARKS (200 QUERIES)
# ============================================================================
st.subheader(f"3. So sánh Tốc độ & Recall@5 thực đo ({benchmarks.get('n_queries', 200)} truy vấn)")

speed_rows = benchmarks.get("full_vs_kmeans_speed", [])
if speed_rows:
    df_speed = pd.DataFrame(speed_rows)
    sc1, sc2 = st.columns([1.25, 1.0])

    with sc1:
        st.dataframe(
            df_speed.rename(columns={
                "method": "Phương pháp",
                "avg_candidates": "Số file quét TB",
                "avg_query_ms": "Avg time/query (ms)",
                "recall_at_5": "Recall@5 (cùng loài)",
                "overlap_with_full_at_5": "Overlap@5 vs Full",
                "speedup_vs_full": "Tốc độ so với Full (×)",
            }).drop(columns=["k"]),
            width="stretch",
            hide_index=True,
        )

    with sc2:
        tradeoff_chart = (
            alt.Chart(df_speed)
            .mark_circle(size=120)
            .encode(
                x=alt.X("speedup_vs_full:Q", title="Tốc độ so với Full-scan (×)"),
                y=alt.Y("recall_at_5:Q", title="Recall@5 (cùng loài)", scale=alt.Scale(zero=False)),
                color=alt.Color("method:N", title="Phương pháp"),
                tooltip=[
                    "method:N",
                    alt.Tooltip("avg_query_ms:Q", title="Avg ms", format=".4f"),
                    alt.Tooltip("recall_at_5:Q", title="Recall@5", format=".4f"),
                    alt.Tooltip("speedup_vs_full:Q", title="Speedup ×", format=".2f"),
                ],
            )
            .properties(height=280, title="Đánh đổi Tốc độ (×) vs Recall@5")
        )
        st.altair_chart(tradeoff_chart, width="stretch")

algo_rows = benchmarks.get("knn_algorithms", [])
strat_rows = benchmarks.get("four_strategies", [])

sub_a, sub_b = st.columns(2)
with sub_a:
    st.markdown("#### So sánh thuật toán KNN (Mục 3.4)")
    if algo_rows:
        df_algo = pd.DataFrame(algo_rows)
        st.dataframe(
            df_algo.rename(columns={
                "algorithm": "Thuật toán",
                "fit_time_ms": "Fit time (ms)",
                "avg_query_ms": "Avg query (ms)",
                "relative_to_kdtree": "So với KDTree (×)",
            })[["Thuật toán", "Fit time (ms)", "Avg query (ms)", "So với KDTree (×)"]],
            width="stretch",
            hide_index=True,
        )
        algo_bar = (
            alt.Chart(df_algo)
            .mark_bar(cornerRadiusEnd=4, color="#2563EB")
            .encode(
                x=alt.X("avg_query_ms:Q", title="Thời gian truy vấn TB (ms)"),
                y=alt.Y("algorithm:N", sort="x", title="Thuật toán"),
                tooltip=["algorithm:N", alt.Tooltip("avg_query_ms:Q", format=".4f")],
            )
            .properties(height=160)
        )
        st.altair_chart(algo_bar, width="stretch")

with sub_b:
    st.markdown("#### So sánh 4 chiến lược truy xuất (Mục 8.4)")
    if strat_rows:
        df_strat = pd.DataFrame(strat_rows)
        st.dataframe(
            df_strat.rename(columns={
                "strategy": "Chiến lược",
                "avg_candidates": "File quét TB",
                "avg_query_ms": "Avg query (ms)",
                "recall_at_5": "Recall@5",
                "speedup_vs_full": "Speedup (×)",
            }),
            width="stretch",
            hide_index=True,
        )

st.divider()

# ============================================================================
# SECTION 4: EXPORTABLE STATIC REPORT FIGURES
# ============================================================================
st.subheader("4. Biểu đồ tĩnh chuẩn báo cáo (PNG độ phân giải cao)")
img_tabs = st.tabs(["Elbow & Silhouette", "PCA 2D Scatter", "Heatmap Loài × Cụm"])

for tab, img_path, caption in zip(
    img_tabs,
    [ELBOW_SILHOUETTE_PNG, PCA_SCATTER_PNG, SPECIES_CLUSTER_HEATMAP_PNG],
    [
        "Biểu đồ Elbow (Inertia), Silhouette và Davies-Bouldin theo K (2..20)",
        "Phép chiếu PCA 2D: 13 cụm K-Means (trái) so với 13 loài thực tế (phải)",
        "Ma trận phân bố 13 loài × 13 cụm K-Means trên 648 tệp âm thanh",
    ],
):
    with tab:
        if img_path.is_file():
            st.image(str(img_path), caption=caption, width="stretch")
            st.download_button(
                label=f"⬇️ Tải ảnh {img_path.name}",
                data=img_path.read_bytes(),
                file_name=img_path.name,
                mime="image/png",
            )
        else:
            st.warning(f"Chưa tìm thấy `{img_path.name}`. Hãy chạy `python scripts/run_benchmark.py`.")
