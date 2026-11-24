"""Streamlit main entry point — Home & System Overview."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_app.config import (
    DELTA_WIDTH,
    FEATURE_DIM,
    HOP_LENGTH,
    KMEANS_N_CLUSTERS,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
    WIN_LENGTH,
)
from web_app.ui_components import (
    format_ms,
    get_pipeline,
    inject_custom_css,
    load_model_metadata,
)

st.set_page_config(
    page_title="Animal Sound AI — K-Means + KNN",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

st.title("🐾 Hệ Thống Tìm Kiếm Âm Thanh Động Vật Tương Đồng")
st.markdown(
    "**Phân cụm K-Means (K=13) kết hợp Tìm kiếm Láng giềng gần nhất KNN (KDTree) trên Vector đặc trưng âm thanh 92 chiều (Bộ C)**"
)

try:
    pipeline = get_pipeline()
    metadata = load_model_metadata()
except FileNotFoundError as exc:
    st.error(
        f"Chưa tìm thấy tệp dữ liệu hoặc mô hình đã huấn luyện: `{exc}`\n\n"
        "Hãy chạy quy trình chuẩn bị dữ liệu và huấn luyện offline:\n"
        "```bash\n"
        "python scripts/prepare_data.py\n"
        "python scripts/train_models.py\n"
        "python scripts/run_benchmark.py\n"
        "```"
    )
    st.stop()

# Navigation quick links
st.markdown("### Điều hướng nhanh")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.page_link("pages/1_search.py", label="🔍 Trang 1: Tìm kiếm & So sánh 2 bản", width="stretch")
with nav2:
    st.page_link("pages/2_report.py", label="📊 Trang 2: Báo cáo Kỹ thuật & Benchmark", width="stretch")
with nav3:
    st.page_link("pages/3_clusters.py", label="🧭 Trang 3: Trực quan hóa Cụm (PCA / t-SNE)", width="stretch")

st.divider()

# Top KPI Cards
ds_info = metadata.get("dataset", {})
km_metrics = metadata.get("kmeans_metrics", pipeline.kmeans_engine.metrics)
init_timings = metadata.get("init_timings_ms", {})
speed_rows = metadata.get("benchmarks", {}).get("full_vs_kmeans_speed", [])
k13_speed = next((r for r in speed_rows if r.get("k") == KMEANS_N_CLUSTERS), None)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tổng số tệp WAV", f"{ds_info.get('n_files', len(pipeline.df_zscore)):,}")
c2.metric("Số loài động vật", f"{ds_info.get('n_species', pipeline.df_zscore['species'].nunique())}")
c3.metric("Chiều vector (Bộ C)", f"{FEATURE_DIM}D (Z-score)")
c4.metric("Số cụm K-Means", f"K = {KMEANS_N_CLUSTERS}")
c5.metric(
    "Tốc độ K=13 vs Full",
    f"{k13_speed['speedup_vs_full']:.2f}×" if k13_speed else "Đã huấn luyện",
    delta=f"Recall@5: {k13_speed['recall_at_5']:.3f}" if k13_speed else None,
)

# Pipeline Overview & Feature Breakdown
left_col, right_col = st.columns([1.15, 1.0])

with left_col:
    st.subheader("Kiến trúc Pipeline hệ thống")
    st.code(
        """[Upload WAV / MP3 / FLAC / OGG]
        │
        ▼
[Trích xuất 92D vector]   ← librosa (SR=16000, win=400, hop=160, n_fft=512)
        │
        ▼
[Chuẩn hóa Z-score]       ← dùng μ, σ đóng băng từ zscore_scaler_params.json
        │                   (KHÔNG fit lại scaler trên file upload)
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
[BẢN 1: CÓ K-MEANS (K=13)]              [BẢN 2: KHÔNG K-MEANS (Full-scan)]
        │                                             │
        ▼                                             ▼
[KMeans.predict() → cụm C]              [KDTree trên toàn bộ 648 file]
        │                                             │
        ▼                                             │
[KNN chỉ trên cụm C (~50 file)]                       │
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
          [Top-5 kết quả + khoảng cách + thông số báo cáo kỹ thuật]""",
        language="text",
    )

with right_col:
    st.subheader("Cấu trúc Vector 92 chiều (Bộ C)")
    feat_struct_df = pd.DataFrame([
        {"Nhóm đặc trưng": "Bộ A – Đặc trưng cơ bản", "Số chiều": 14, "Chi tiết": "rms, zcr, centroid, bandwidth, rolloff, flatness, flux × (mean, std)"},
        {"Nhóm đặc trưng": "Bộ B – MFCC", "Số chiều": 26, "Chi tiết": "13 hệ số MFCC × (mean, std)"},
        {"Nhóm đặc trưng": "Delta MFCC (bậc 1)", "Số chiều": 26, "Chi tiết": "13 hệ số Delta × (mean_abs, std)"},
        {"Nhóm đặc trưng": "Delta-Delta MFCC (bậc 2)", "Số chiều": 26, "Chi tiết": "13 hệ số Delta² × (mean_abs, std)"},
        {"Nhóm đặc trưng": "Tổng cộng (Bộ C)", "Số chiều": 92, "Chi tiết": "Chuẩn hóa Z-score với μ, σ trên 648 tệp gốc"},
    ])
    st.dataframe(feat_struct_df, width="stretch", hide_index=True)

    st.subheader("Cấu hình xử lý tín hiệu (DSP)")
    dsp_df = pd.DataFrame([
        {"Tham số": "Sample rate chuẩn", "Giá trị": f"{SAMPLE_RATE:,} Hz (Mono)"},
        {"Tham số": "Window length", "Giá trị": f"{WIN_LENGTH} mẫu (25 ms, cửa sổ Hann)"},
        {"Tham số": "Hop length", "Giá trị": f"{HOP_LENGTH} mẫu (10 ms)"},
        {"Tham số": "N_FFT / N_MFCC / Delta width", "Giá trị": f"{N_FFT} / {N_MFCC} / {DELTA_WIDTH}"},
    ])
    st.dataframe(dsp_df, width="stretch", hide_index=True)

st.divider()

# Dataset Species Distribution & Model Metrics
col_dist, col_status = st.columns([1.2, 1.0])

with col_dist:
    st.subheader("Phân bố 648 tệp âm thanh theo 13 loài")
    sp_counts = (
        pipeline.df_zscore["species"]
        .value_counts()
        .reset_index()
    )
    sp_counts.columns = ["Loài", "Số tệp"]
    chart_sp = (
        alt.Chart(sp_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2563EB")
        .encode(
            x=alt.X("Loài:N", sort="-y", title="Loài động vật"),
            y=alt.Y("Số tệp:Q", title="Số lượng tệp WAV"),
            tooltip=["Loài:N", "Số tệp:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart_sp, width="stretch")

with col_status:
    st.subheader("Trạng thái mô hình đã huấn luyện (K=13)")
    metrics_df = pd.DataFrame([
        {"Chỉ số": "Silhouette Score ↑", "Giá trị": f"{km_metrics.get('silhouette', 0.0):.4f}"},
        {"Chỉ số": "Davies-Bouldin Index ↓", "Giá trị": f"{km_metrics.get('davies_bouldin', 0.0):.4f}"},
        {"Chỉ số": "Calinski-Harabasz Index ↑", "Giá trị": f"{km_metrics.get('calinski_harabasz', 0.0):.2f}"},
        {"Chỉ số": "Adjusted Rand Index (ARI) ↑", "Giá trị": f"{km_metrics.get('ari', 0.0):.4f}"},
        {"Chỉ số": "Normalized Mutual Info (NMI) ↑", "Giá trị": f"{km_metrics.get('nmi', 0.0):.4f}"},
        {"Chỉ số": "Purity ↑", "Giá trị": f"{km_metrics.get('purity', 0.0):.4f}"},
    ])
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    if init_timings:
        st.caption(
            f"Khởi tạo offline — Load CSV+Scaler: `{format_ms(init_timings.get('load_csv_and_scaler_ms', 0.0))}` | "
            f"KMeans fit (K=13): `{format_ms(init_timings.get('kmeans_fit_ms', 0.0))}` | "
            f"KDTree fit (648×92): `{format_ms(init_timings.get('kdtree_fit_ms', 0.0))}`"
        )
