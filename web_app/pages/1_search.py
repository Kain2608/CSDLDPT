"""Page 1 — Audio Similarity Search (Full-scan vs K-Means K=13) with full Section 9 diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.config import KNN_DEFAULT_TOP_K, SUPPORTED_AUDIO_EXTENSIONS, resolve_audio_path
from web_app.engine.feature_engine import get_feature_group
from web_app.ui_components import (
    build_before_after_zscore_chart,
    build_cluster_species_bar,
    build_vector_92d_heatmap,
    format_bytes,
    format_ms,
    get_pipeline,
    inject_custom_css,
    render_knn_results_panel,
)

st.set_page_config(
    page_title="Tìm kiếm âm thanh tương đồng",
    page_icon="🔍",
    layout="wide",
)
inject_custom_css()

st.title("🔍 Tìm Kiếm Âm Thanh Tương Đồng")
st.markdown("Upload 1 tệp âm thanh (WAV/MP3/FLAC/OGG) hoặc chọn tệp mẫu để trích xuất 92D, chuẩn hóa Z-score và so sánh **Có K-Means (K=13)** với **Không K-Means (Full-scan)**.")

pipeline = get_pipeline()

# --- Input & Controls ---
with st.sidebar:
    st.header("Cấu hình tìm kiếm")
    backend_label = st.radio(
        "Nguồn truy vấn KNN",
        options=[
            "PostgreSQL pgvector (SQL)",
            "Python KDTree (RAM)",
        ],
        index=0,
    )
    backend = "pgvector" if backend_label.startswith("PostgreSQL") else "kdtree"

    mode_label = st.radio(
        "Chế độ chạy",
        options=[
            "So sánh cả hai",
            "Có K-means (K=13)",
            "Không K-means (Full-scan)",
        ],
        index=0,
    )
    mode_map = {
        "So sánh cả hai": "compare",
        "Có K-means (K=13)": "with_kmeans",
        "Không K-means (Full-scan)": "without_kmeans",
    }
    mode = mode_map[mode_label]

    top_k = st.slider("Số láng giềng trả về (Top-K)", min_value=1, max_value=15, value=KNN_DEFAULT_TOP_K)

    with st.expander("Tùy chọn nâng cao (Mục 3.2 & Mục 8)", expanded=False):
        feature_set_label = st.selectbox(
            "Bộ đặc trưng (PostgreSQL pgvector)",
            [
                "C — Toàn bộ 92 chiều (Mặc định)",
                "A — 14 chiều cơ bản",
                "B — 78 chiều MFCC + Delta + Delta²",
            ],
            index=0,
            disabled=(backend != "pgvector"),
        )
        feature_set = feature_set_label[0]
        knn_algo = st.selectbox("Thuật toán KNN (bản Full-scan)", ["kd_tree", "ball_tree", "brute"], index=0)
        metric = st.selectbox("Độ đo khoảng cách (Metric)", ["euclidean", "cosine"], index=0)
        n_probe_clusters = st.slider(
            "Số cụm tìm kiếm (K-means multi-cluster)",
            min_value=1,
            max_value=3,
            value=1,
            help="Mặc định = 1 (KNN trong cụm gần nhất). Đặt 2 hoặc 3 để thử chiến lược đa cụm (Mục 8.3).",
        )
        run_4_strategies = st.checkbox(
            "So sánh thêm 4 chiến lược (Mục 8.4: Full, Top-1 cụm, Top-2 cụm, PCA 92→20)",
            value=False,
        )

input_tab_upload, input_tab_sample = st.tabs(["📤 Upload file âm thanh", "🎵 Chọn file mẫu từ Dataset (648 file)"])

uploaded_bytes: bytes | None = None
uploaded_name: str | None = None
selected_sample_idx: int | None = None
selected_sample_audio_path: Path | None = None
trigger_search = False

with input_tab_upload:
    uploaded_file = st.file_uploader(
        "Chọn tệp âm thanh đầu vào (WAV, MP3, FLAC, OGG)",
        type=list(SUPPORTED_AUDIO_EXTENSIONS),
    )
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        uploaded_name = uploaded_file.name
        st.audio(uploaded_bytes)
        if st.button("🚀 Tìm kiếm từ file upload", type="primary", key="btn_search_upload"):
            trigger_search = True
            st.session_state["last_query_source"] = ("upload", uploaded_bytes, uploaded_name)

with input_tab_sample:
    species_list = sorted(pipeline.df_zscore["species"].unique().tolist())
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        chosen_species = st.selectbox("Lọc theo loài", ["Tất cả"] + species_list)
    sub_df = (
        pipeline.df_zscore
        if chosen_species == "Tất cả"
        else pipeline.df_zscore[pipeline.df_zscore["species"] == chosen_species]
    )
    options = sub_df.index.tolist()
    with col_s2:
        selected_sample_idx = st.selectbox(
            "Chọn tệp âm thanh mẫu",
            options=options,
            format_func=lambda idx: f"{pipeline.df_zscore.loc[idx, 'name']} — Loài: {pipeline.df_zscore.loc[idx, 'species']} (Cụm #{pipeline.df_zscore.loc[idx, 'cluster_id']})",
        )
    if selected_sample_idx is not None:
        sample_row = pipeline.df_zscore.loc[selected_sample_idx]
        selected_sample_audio_path = resolve_audio_path(str(sample_row["path"]))
        if selected_sample_audio_path is not None:
            st.audio(str(selected_sample_audio_path))
        if st.button("🚀 Tìm kiếm bằng file mẫu này", type="primary", key="btn_search_sample"):
            trigger_search = True
            st.session_state["last_query_source"] = ("sample", int(selected_sample_idx), str(sample_row["name"]))

# Keep previous query active when user adjusts sidebar controls
active_source = st.session_state.get("last_query_source")
if not trigger_search and active_source is None:
    st.info("👆 Hãy upload một tệp âm thanh hoặc chọn tệp mẫu ở tab bên trên rồi bấm nút **Tìm kiếm**.")
    st.stop()

source_type, source_payload, source_name = active_source
try:
    with st.spinner(f"Đang trích xuất đặc trưng 92D và tìm kiếm cho `{source_name}`..."):
        if source_type == "upload":
            qres = pipeline.query_audio_bytes(
                audio_bytes=source_payload,
                filename=source_name,
                mode=mode,
                top_k=top_k,
                metric=metric,
                algorithm=knn_algo,
                n_probe_clusters=n_probe_clusters,
                include_advanced_strategies=run_4_strategies,
                backend=backend,
                feature_set=feature_set,
            )
        else:
            qres = pipeline.query_dataset_sample(
                row_index=int(source_payload),
                mode=mode,
                top_k=top_k,
                metric=metric,
                algorithm=knn_algo,
                n_probe_clusters=n_probe_clusters,
                include_advanced_strategies=run_4_strategies,
                backend=backend,
                feature_set=feature_set,
            )
except ValueError as err:
    st.error(f"Tệp âm thanh không hợp lệ: {err}")
    st.stop()
except Exception as err:
    st.error(f"Lỗi khi xử lý tệp âm thanh `{source_name}`: {err}")
    st.stop()

ext_meta = qres.extraction.metadata
scaling = qres.scaling
km_pred = qres.kmeans_prediction

st.divider()

# ============================================================================
# 9.5 THÔNG SỐ TỐC ĐỘ (END-TO-END TIMING STRIP)
# ============================================================================
st.subheader("⏱️ 1. Thông số Tốc độ xử lý (ms)")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Trích xuất 92D", format_ms(ext_meta.extraction_time_ms))
t2.metric("Chuẩn hóa Z-score", format_ms(scaling.scaling_time_ms))
t3.metric("K-Means predict", format_ms(km_pred.predict_time_ms))
if qres.result_with_kmeans is not None and qres.result_without_kmeans is not None:
    t4.metric(
        "KNN (Có vs Không KM)",
        format_ms(qres.result_with_kmeans.search_time_ms),
        delta=f"Full: {format_ms(qres.result_without_kmeans.search_time_ms)}",
        delta_color="off",
    )
elif qres.result_with_kmeans is not None:
    t4.metric("Tìm kiếm KNN (Cụm)", format_ms(qres.result_with_kmeans.search_time_ms))
else:
    t4.metric("Tìm kiếm KNN (Full)", format_ms(qres.result_without_kmeans.search_time_ms))
t5.metric("Tổng End-to-End", format_ms(qres.total_end_to_end_ms))

# ============================================================================
# 9.1 THÔNG SỐ FILE ĐẦU VÀO
# ============================================================================
st.subheader("🎧 2. Thông số tệp âm thanh đầu vào")
info_col, wave_col = st.columns([1.15, 1.0])

with info_col:
    file_info_df = pd.DataFrame([
        {"Thuộc tính": "Tên tệp", "Giá trị": ext_meta.filename},
        {"Thuộc tính": "Định dạng / Dung lượng", "Giá trị": f"{ext_meta.file_format} | {format_bytes(ext_meta.file_size_bytes)} ({ext_meta.file_size_bytes:,} bytes)"},
        {"Thuộc tính": "Sample rate gốc → Sau resample", "Giá trị": f"{ext_meta.original_sample_rate:,} Hz ({ext_meta.original_channels} ch) → {ext_meta.resampled_sample_rate:,} Hz (Mono)"},
        {"Thuộc tính": "Thời lượng (giây)", "Giá trị": f"{ext_meta.duration_sec:.4f} s"},
        {"Thuộc tính": "Số mẫu (samples @ 16 kHz)", "Giá trị": f"{ext_meta.num_samples:,} mẫu"},
        {"Thuộc tính": "Số khung (hop=160) / Khung STFT", "Giá trị": f"{ext_meta.num_frames_hop:,} khung (len/160) | {ext_meta.num_stft_frames:,} khung STFT"},
    ])
    st.dataframe(file_info_df, width="stretch", hide_index=True)

with wave_col:
    waveform = qres.extraction.waveform
    if len(waveform) > 0:
        step = max(1, len(waveform) // 600)
        wf_sub = waveform[::step]
        t_axis = np.arange(len(wf_sub)) * (step / float(ext_meta.resampled_sample_rate))
        wf_df = pd.DataFrame({"Thời gian (s)": t_axis, "Biên độ": wf_sub})
        wf_chart = (
            alt.Chart(wf_df)
            .mark_line(color="#2563EB", strokeWidth=1.2)
            .encode(
                x=alt.X("Thời gian (s):Q"),
                y=alt.Y("Biên độ:Q", scale=alt.Scale(domain=[-1.05, 1.05])),
            )
            .properties(height=230, title="Dạng sóng tín hiệu sau resample (16 kHz Mono)")
        )
        st.altair_chart(wf_chart, width="stretch")
    else:
        st.info("Tệp mẫu đang dùng trực tiếp vector 92D từ CSV (không có dạng sóng WAV gốc trên máy).")

st.divider()

# ============================================================================
# 9.4 & 9.6 KẾT QUẢ TOP-K VÀ SO SÁNH 2 BẢN (SIDE-BY-SIDE)
# ============================================================================
st.subheader(f"🏆 3. Kết quả Top-{top_k} âm thanh tương đồng nhất")

if qres.comparison_summary is not None and qres.result_without_kmeans is not None and qres.result_with_kmeans is not None:
    comp = qres.comparison_summary
    res_no = qres.result_without_kmeans
    res_km = qres.result_with_kmeans

    top1_no = f"{res_no.matches[0].name} ({res_no.matches[0].species}, d={res_no.matches[0].distance:.4f})"
    top1_km = f"{res_km.matches[0].name} ({res_km.matches[0].species}, d={res_km.matches[0].distance:.4f})"
    top5_no = ", ".join(m.name for m in res_no.matches)
    top5_km = ", ".join(m.name for m in res_km.matches)

    comp_df = pd.DataFrame([
        {
            "Chỉ tiêu so sánh": "Thời gian tìm kiếm KNN",
            "Không K-means (Full-scan)": format_ms(comp["full_search_ms"]),
            f"Có K-means (K={km_pred.n_clusters})": format_ms(comp["kmeans_search_ms"]),
        },
        {
            "Chỉ tiêu so sánh": "Tổng thời gian truy xuất (KMeans.predict + KNN)",
            "Không K-means (Full-scan)": format_ms(res_no.total_retrieval_time_ms),
            f"Có K-means (K={km_pred.n_clusters})": format_ms(comp["kmeans_total_retrieval_ms"]),
        },
        {
            "Chỉ tiêu so sánh": "Số file quét",
            "Không K-means (Full-scan)": f"{comp['full_candidates']} file (100%)",
            f"Có K-means (K={km_pred.n_clusters})": f"{comp['kmeans_candidates']} file (giảm {comp['candidate_reduction_pct']:.1f}%)",
        },
        {
            "Chỉ tiêu so sánh": "Kết quả Top-1",
            "Không K-means (Full-scan)": top1_no,
            f"Có K-means (K={km_pred.n_clusters})": top1_km,
        },
        {
            "Chỉ tiêu so sánh": f"Danh sách Top-{top_k}",
            "Không K-means (Full-scan)": top5_no,
            f"Có K-means (K={km_pred.n_clusters})": top5_km,
        },
        {
            "Chỉ tiêu so sánh": f"Độ trùng khớp Top-{top_k} so với Full-scan",
            "Không K-means (Full-scan)": f"{top_k}/{top_k} (100%)",
            f"Có K-means (K={km_pred.n_clusters})": f"{comp['overlap_count']}/{top_k} ({comp['overlap_ratio']*100:.1f}%)",
        },
        {
            "Chỉ tiêu so sánh": "Tốc độ tương đối (KNN search)",
            "Không K-means (Full-scan)": "1.00×",
            f"Có K-means (K={km_pred.n_clusters})": f"{comp['speedup_knn_only']:.2f}× (Tổng truy xuất: {comp['speedup_total_retrieval']:.2f}×)",
        },
    ])
    st.markdown("#### Bảng so sánh 2 bản (Side-by-Side — Mục 9.6)")
    st.dataframe(comp_df, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        render_knn_results_panel(res_no, accent="blue")
    with col_right:
        render_knn_results_panel(res_km, accent="green")

elif qres.result_with_kmeans is not None:
    render_knn_results_panel(qres.result_with_kmeans, accent="green")
elif qres.result_without_kmeans is not None:
    render_knn_results_panel(qres.result_without_kmeans, accent="blue")

# Optional 4-strategy comparison (Section 8.4)
if run_4_strategies and qres.result_multi_cluster is not None and qres.result_pca_kmeans is not None:
    st.divider()
    st.subheader("🧪 So sánh 4 chiến lược truy xuất (Mục 8.4)")
    s_rows = []
    for res_item in [
        qres.result_without_kmeans,
        qres.result_with_kmeans,
        qres.result_multi_cluster,
        qres.result_pca_kmeans,
    ]:
        if res_item is None:
            continue
        s_rows.append({
            "Chiến lược": res_item.mode_name,
            "Số file quét": res_item.candidates_scanned,
            "Cụm tìm kiếm": str(res_item.clusters_searched) if len(res_item.clusters_searched) <= 4 else "Tất cả (13 cụm)",
            "Thời gian KNN (ms)": round(res_item.search_time_ms, 4),
            "Tổng truy xuất (ms)": round(res_item.total_retrieval_time_ms, 4),
            "Top-1": f"{res_item.matches[0].name} ({res_item.matches[0].species}, d={res_item.matches[0].distance:.4f})",
            "Dự đoán loài (Weighted)": res_item.predicted_species_weighted,
        })
    st.dataframe(pd.DataFrame(s_rows), width="stretch", hide_index=True)

st.divider()

# ============================================================================
# 9.3 THÔNG SỐ K-MEANS
# ============================================================================
st.subheader("🎯 4. Thông số phân cụm K-Means (Mục 9.3)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Số cụm K", f"K = {km_pred.n_clusters}")
k2.metric("Cụm được gán", f"Cụm #{km_pred.cluster_id}")
k3.metric("Khoảng cách tới Centroid", f"{km_pred.distance_to_centroid:.4f}")
k4.metric("Số file trong cụm", f"{km_pred.cluster_size} / 648")

km_col1, km_col2 = st.columns([1.15, 1.0])
with km_col1:
    st.altair_chart(
        build_cluster_species_bar(km_pred.species_distribution, km_pred.cluster_id),
        width="stretch",
    )

with km_col2:
    st.markdown("**Khoảng cách Euclidean từ file query đến 13 Centroids**")
    cent_rows = [
        {
            "Hạng": rank,
            "Cụm": f"Cụm #{c_id}",
            "Khoảng cách tới Centroid": round(float(km_pred.all_centroid_distances[c_id]), 4),
            "Số file": int(len(pipeline.kmeans_engine.cluster_members[c_id])),
            "Loài chủ đạo": next(iter(pipeline.kmeans_engine.cluster_species_dist[c_id].keys()), "N/A"),
        }
        for rank, c_id in enumerate(km_pred.nearest_clusters_order, start=1)
    ]
    st.dataframe(pd.DataFrame(cent_rows), width="stretch", hide_index=True, height=250)

km_m = pipeline.kmeans_engine.metrics
with st.expander("Chỉ số chất lượng mô hình K-Means (K=13) & Vector Centroid 92 chiều", expanded=False):
    m_cols = st.columns(6)
    m_cols[0].metric("Silhouette ↑", f"{km_m.get('silhouette', 0.0):.4f}")
    m_cols[1].metric("ARI ↑", f"{km_m.get('ari', 0.0):.4f}")
    m_cols[2].metric("NMI ↑", f"{km_m.get('nmi', 0.0):.4f}")
    m_cols[3].metric("Davies-Bouldin ↓", f"{km_m.get('davies_bouldin', 0.0):.4f}")
    m_cols[4].metric("Calinski-Harabasz ↑", f"{km_m.get('calinski_harabasz', 0.0):.2f}")
    m_cols[5].metric("Purity ↑", f"{km_m.get('purity', 0.0):.4f}")

    cent_df = pd.DataFrame({
        "#": np.arange(1, len(scaling.feature_names) + 1),
        "Đặc trưng": scaling.feature_names,
        "Nhóm": [get_feature_group(f) for f in scaling.feature_names],
        f"Centroid Cụm #{km_pred.cluster_id}": np.round(km_pred.centroid, 6),
        "Query (Z-score)": np.round(scaling.vector_zscore, 6),
        "|Query - Centroid|": np.round(np.abs(scaling.vector_zscore - km_pred.centroid), 6),
    })
    st.dataframe(cent_df, width="stretch", hide_index=True, height=280)

st.divider()

# ============================================================================
# 9.2 THÔNG SỐ VECTOR 92 CHIỀU & CHUẨN HÓA Z-SCORE (μ, σ)
# ============================================================================
st.subheader("🧬 5. Chi tiết Vector 92 chiều & Chuẩn hóa Z-Score (Mục 9.2)")
vec_tab1, vec_tab2, vec_tab3 = st.tabs([
    "Heatmap 92 chiều (Z-score)",
    "So sánh Trước / Sau Z-score",
    "Bảng đầy đủ 92 giá trị (Raw, μ, σ, Z-score)",
])

with vec_tab1:
    st.altair_chart(
        build_vector_92d_heatmap(scaling.vector_zscore, scaling.feature_names),
        width="stretch",
    )

with vec_tab2:
    group_choice = st.selectbox(
        "Chọn nhóm đặc trưng để so sánh trực quan Trước / Sau Z-score",
        [
            "Bộ A - Cơ bản (14D)",
            "Bộ B - MFCC (26D)",
            "Delta MFCC (26D)",
            "Delta-Delta MFCC (26D)",
        ],
    )
    st.altair_chart(
        build_before_after_zscore_chart(
            scaling.vector_raw,
            scaling.vector_zscore,
            scaling.feature_names,
            selected_group=group_choice,
        ),
        width="stretch",
    )

with vec_tab3:
    full_vec_df = pd.DataFrame({
        "#": np.arange(1, len(scaling.feature_names) + 1),
        "Đặc trưng": scaling.feature_names,
        "Nhóm": [get_feature_group(f) for f in scaling.feature_names],
        "Giá trị gốc (x)": np.round(scaling.vector_raw, 6),
        "Mean gốc (μ)": np.round(scaling.mean, 6),
        "Std gốc (σ)": np.round(scaling.std, 6),
        "Sau Z-score (z = (x-μ)/σ)": np.round(scaling.vector_zscore, 6),
    })
    st.dataframe(full_vec_df, width="stretch", hide_index=True, height=380)
