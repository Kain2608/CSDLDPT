"""Page 3 — Interactive Cluster Visualization (PCA / t-SNE 2D, Cluster Sizes, Species Heatmap, Inspector)."""

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

from web_app.config import KMEANS_N_CLUSTERS, resolve_audio_path
from web_app.ui_components import (
    build_cluster_species_bar,
    get_pipeline,
    inject_custom_css,
    load_model_metadata,
    load_projections,
)

st.set_page_config(
    page_title="Trực quan hóa Cụm K-Means",
    page_icon="🧭",
    layout="wide",
)
inject_custom_css()

st.title("🧭 Trực Quan Hóa Cụm K-Means (K = 13)")
st.markdown("Khám phá không gian vector 92 chiều qua phép chiếu **PCA 2D** và **t-SNE 2D**, phân bố kích thước từng cụm và ma trận **Loài × Cụm**.")

pipeline = get_pipeline()
metadata = load_model_metadata()
proj_df, cent_df = load_projections()

if proj_df.empty:
    st.error("Chưa có dữ liệu tọa độ PCA/t-SNE (`projections_2d.csv`). Hãy chạy `python scripts/train_models.py`.")
    st.stop()

proj_df = proj_df.copy()
proj_df["cluster_label"] = proj_df["cluster_id"].apply(lambda c: f"Cụm {int(c):02d}")

# --- Controls ---
with st.sidebar:
    st.header("Tùy chọn hiển thị")
    projection_method = st.radio("Phép chiếu 2D", ["PCA 2D", "t-SNE 2D"], index=0)
    color_by = st.radio("Tô màu điểm dữ liệu theo", ["Cụm K-Means (K=13)", "Loài thực tế (13 loài)"], index=0)
    show_centroids = st.checkbox("Hiển thị tâm cụm (Centroids) trên PCA 2D", value=True)
    selected_species_filter = st.multiselect(
        "Lọc theo loài (trống = tất cả)",
        options=sorted(proj_df["species"].unique().tolist()),
        default=[],
    )

filtered_df = (
    proj_df[proj_df["species"].isin(selected_species_filter)]
    if selected_species_filter
    else proj_df
)

# ============================================================================
# 1. INTERACTIVE 2D SCATTER (PCA / t-SNE)
# ============================================================================
st.subheader(f"1. Biểu đồ phân tán {projection_method} ({len(filtered_df)} tệp)")

x_col = "pca_x" if projection_method == "PCA 2D" else "tsne_x"
y_col = "pca_y" if projection_method == "PCA 2D" else "tsne_y"
color_field = "cluster_label:N" if color_by.startswith("Cụm") else "species:N"
color_title = "Cụm K-Means" if color_by.startswith("Cụm") else "Loài động vật"

pca_var = metadata.get("pca_2d_explained_variance_ratio", [])
x_title = f"PC1 ({pca_var[0]*100:.1f}% var)" if projection_method == "PCA 2D" and len(pca_var) == 2 else "t-SNE Dim 1"
y_title = f"PC2 ({pca_var[1]*100:.1f}% var)" if projection_method == "PCA 2D" and len(pca_var) == 2 else "t-SNE Dim 2"

scatter = (
    alt.Chart(filtered_df)
    .mark_circle(size=55, opacity=0.82)
    .encode(
        x=alt.X(f"{x_col}:Q", title=x_title),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        color=alt.Color(color_field, scale=alt.Scale(scheme="category20"), title=color_title),
        tooltip=[
            alt.Tooltip("name:N", title="Tên tệp"),
            alt.Tooltip("species:N", title="Loài"),
            alt.Tooltip("cluster_label:N", title="Cụm"),
            alt.Tooltip("duration:Q", title="Thời lượng (s)", format=".3f"),
        ],
    )
    .interactive()
)

if projection_method == "PCA 2D" and show_centroids and not cent_df.empty:
    cent_plot = cent_df.copy()
    cent_plot["label"] = cent_plot["cluster_id"].apply(lambda c: f"C{int(c)}")
    cent_points = (
        alt.Chart(cent_plot)
        .mark_point(shape="cross", size=220, strokeWidth=3, color="#111827", filled=False)
        .encode(
            x="pca_x:Q",
            y="pca_y:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Tâm cụm"),
                alt.Tooltip("size:Q", title="Số file trong cụm"),
            ],
        )
    )
    cent_text = (
        alt.Chart(cent_plot)
        .mark_text(dy=-12, fontSize=11, fontWeight="bold", color="#111827")
        .encode(x="pca_x:Q", y="pca_y:Q", text="label:N")
    )
    final_scatter = (scatter + cent_points + cent_text).properties(height=470)
else:
    final_scatter = scatter.properties(height=470)

st.altair_chart(final_scatter, width="stretch")

st.divider()

# ============================================================================
# 2. CLUSTER SIZE BAR CHART & SPECIES x CLUSTER HEATMAP
# ============================================================================
col_bar, col_heat = st.columns([1.0, 1.25])

with col_bar:
    st.subheader("2. Số lượng tệp WAV theo từng cụm")
    size_rows = []
    for c in range(KMEANS_N_CLUSTERS):
        dist = pipeline.kmeans_engine.cluster_species_dist[c]
        dom_sp, dom_cnt = next(iter(dist.items())) if dist else ("N/A", 0)
        sz = int(len(pipeline.kmeans_engine.cluster_members[c]))
        size_rows.append({
            "Cụm": f"C{c:02d}",
            "cluster_id": c,
            "Số file": sz,
            "Loài chủ đạo": f"{dom_sp} ({dom_cnt}/{sz})",
        })
    df_sizes = pd.DataFrame(size_rows)
    avg_size = float(df_sizes["Số file"].mean())

    bar = (
        alt.Chart(df_sizes)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2563EB")
        .encode(
            x=alt.X("Cụm:N", sort=[f"C{c:02d}" for c in range(KMEANS_N_CLUSTERS)], title="Cụm K-Means"),
            y=alt.Y("Số file:Q", title="Số lượng tệp WAV"),
            tooltip=["Cụm:N", "Số file:Q", "Loài chủ đạo:N"],
        )
    )
    avg_rule = (
        alt.Chart(pd.DataFrame({"avg": [avg_size]}))
        .mark_rule(color="#DC2626", strokeDash=[5, 4], strokeWidth=2)
        .encode(y="avg:Q")
    )
    st.altair_chart(
        (bar + avg_rule).properties(height=360, title=f"Kích thước 13 cụm (trung bình {avg_size:.1f} file/cụm)"),
        width="stretch",
    )

with col_heat:
    st.subheader("3. Heatmap phân bố Loài × Cụm")
    ct = pd.crosstab(proj_df["species"], proj_df["cluster_id"]).reset_index()
    melted = ct.melt(id_vars="species", var_name="cluster_id", value_name="count")
    melted["Cụm"] = melted["cluster_id"].apply(lambda c: f"C{int(c):02d}")

    heat_base = alt.Chart(melted).encode(
        x=alt.X("Cụm:N", sort=[f"C{c:02d}" for c in range(KMEANS_N_CLUSTERS)], title="Cụm K-Means"),
        y=alt.Y("species:N", title="Loài động vật"),
    )
    heat_rect = heat_base.mark_rect(stroke="white", strokeWidth=0.6).encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="yellowgreenblue"), title="Số file"),
        tooltip=["species:N", "Cụm:N", alt.Tooltip("count:Q", title="Số tệp WAV")],
    )
    heat_text = heat_base.mark_text(fontSize=10).encode(
        text=alt.condition(alt.datum.count > 0, alt.Text("count:Q"), alt.value("")),
        color=alt.condition(alt.datum.count >= 20, alt.value("white"), alt.value("#111827")),
    )
    st.altair_chart((heat_rect + heat_text).properties(height=380), width="stretch")

st.divider()

# ============================================================================
# 3. INTERACTIVE CLUSTER INSPECTOR
# ============================================================================
st.subheader("4. Khám phá chi tiết từng cụm")
selected_c = st.selectbox(
    "Chọn cụm K-Means muốn kiểm tra",
    options=list(range(KMEANS_N_CLUSTERS)),
    format_func=lambda c: f"Cụm #{c} — {len(pipeline.kmeans_engine.cluster_members[c])} tệp (Chủ đạo: {next(iter(pipeline.kmeans_engine.cluster_species_dist[c].keys()), 'N/A')})",
)

ins_col1, ins_col2 = st.columns([1.0, 1.25])
with ins_col1:
    st.altair_chart(
        build_cluster_species_bar(pipeline.kmeans_engine.cluster_species_dist[selected_c], selected_c),
        width="stretch",
    )

    # Top centroid features (Bộ A - 14D)
    cent_vec = pipeline.kmeans_engine.model.cluster_centers_[selected_c]
    f_names = pipeline.scaler.feature_names
    df_a = pd.DataFrame({
        "Đặc trưng (Bộ A)": f_names[:14],
        "Centroid Z-score": cent_vec[:14],
    })
    c_prof = (
        alt.Chart(df_a)
        .mark_bar()
        .encode(
            x=alt.X("Centroid Z-score:Q", title="Giá trị tâm cụm (Z-score)"),
            y=alt.Y("Đặc trưng (Bộ A):N", sort=None),
            color=alt.condition(alt.datum["Centroid Z-score"] >= 0, alt.value("#2563EB"), alt.value("#DC2626")),
            tooltip=["Đặc trưng (Bộ A):N", alt.Tooltip("Centroid Z-score:Q", format=".4f")],
        )
        .properties(height=280, title=f"Hồ sơ đặc trưng cơ bản (Bộ A) của Tâm Cụm #{selected_c}")
    )
    st.altair_chart(c_prof, width="stretch")

with ins_col2:
    c_members = proj_df[proj_df["cluster_id"] == selected_c][["name", "species", "duration", "path"]].reset_index(drop=True)
    st.markdown(f"**Danh sách {len(c_members)} tệp trong Cụm #{selected_c}**")
    st.dataframe(
        c_members[["name", "species", "duration"]].rename(columns={"name": "Tên tệp", "species": "Loài", "duration": "Thời lượng (s)"}),
        width="stretch",
        hide_index=True,
        height=310,
    )

    if len(c_members) > 0:
        preview_name = st.selectbox("Nghe thử tệp mẫu trong cụm này", options=c_members["name"].tolist())
        row_p = c_members[c_members["name"] == preview_name].iloc[0]
        audio_p = resolve_audio_path(str(row_p["path"]))
        if audio_p is not None:
            st.audio(str(audio_p))
        else:
            st.caption(f"_Không tìm thấy file WAV gốc tại `{row_p['path']}`._")
