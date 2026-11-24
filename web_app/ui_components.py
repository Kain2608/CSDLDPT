"""Reusable Streamlit UI components, cached loaders, and Altair charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from web_app.config import MODEL_METADATA_PATH, MODELS_DIR
from web_app.engine.feature_engine import get_feature_group
from web_app.engine.knn_engine import KNNSearchResult
from web_app.engine.pipeline import AnimalSoundPipeline


def inject_custom_css() -> None:
    """Apply clean, readable custom CSS across all Streamlit pages."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .match-card {
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
        }
        .match-card-km {
            border: 1px solid #E2E8F0;
            border-left: 4px solid #059669;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-species {
            background-color: #DBEAFE;
            color: #1E40AF;
        }
        .badge-cluster {
            background-color: #D1FAE5;
            color: #065F46;
        }
        .badge-dist {
            background-color: #FEF3C7;
            color: #92400E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Đang nạp mô hình K-Means (K=13), KDTree và bộ chuẩn hóa Z-score...")
def get_pipeline() -> AnimalSoundPipeline:
    """Load and cache the full inference pipeline."""
    return AnimalSoundPipeline.load()


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    """Load training & benchmark metadata JSON."""
    if not MODEL_METADATA_PATH.is_file():
        return {}
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_projections() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load precomputed 2D PCA/t-SNE coordinates and cluster centroids."""
    proj_path = MODELS_DIR / "projections_2d.csv"
    cent_path = MODELS_DIR / "centroids_pca_2d.csv"
    proj_df = pd.read_csv(proj_path) if proj_path.is_file() else pd.DataFrame()
    cent_df = pd.read_csv(cent_path) if cent_path.is_file() else pd.DataFrame()
    return proj_df, cent_df


def format_bytes(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_ms(ms: float) -> str:
    """Format milliseconds cleanly."""
    if ms < 1.0:
        return f"{ms:.4f} ms"
    if ms < 100.0:
        return f"{ms:.2f} ms"
    return f"{ms:.1f} ms"


def build_vector_92d_heatmap(vec_zscore: np.ndarray, feature_names: list[str]) -> alt.Chart:
    """Create a 92D Z-score heatmap grouped into 4 feature families (Bộ A, MFCC, Delta, Delta-Delta)."""
    rows = []
    group_pos: dict[str, int] = {}
    for idx, (fname, zval) in enumerate(zip(feature_names, vec_zscore)):
        grp = get_feature_group(fname)
        pos = group_pos.get(grp, 0)
        group_pos[grp] = pos + 1
        rows.append({
            "index": idx + 1,
            "feature": fname,
            "group": grp,
            "pos_in_group": pos + 1,
            "zscore": float(zval),
            "zscore_clipped": float(np.clip(zval, -3.5, 3.5)),
        })
    df = pd.DataFrame(rows)
    group_order = [
        "Bộ A - Cơ bản (14D)",
        "Bộ B - MFCC (26D)",
        "Delta MFCC (26D)",
        "Delta-Delta MFCC (26D)",
    ]
    chart = (
        alt.Chart(df)
        .mark_rect(cornerRadius=2, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X("pos_in_group:O", title="Chiều trong nhóm (1..26)"),
            y=alt.Y("group:N", sort=group_order, title="Nhóm đặc trưng (Bộ C - 92D)"),
            color=alt.Color(
                "zscore_clipped:Q",
                scale=alt.Scale(scheme="redblue", domain=[-3.5, 3.5], reverse=True),
                title="Z-score",
            ),
            tooltip=[
                alt.Tooltip("index:Q", title="Chỉ số (1-92)"),
                alt.Tooltip("feature:N", title="Đặc trưng"),
                alt.Tooltip("group:N", title="Nhóm"),
                alt.Tooltip("zscore:Q", title="Giá trị Z-score", format=".4f"),
            ],
        )
        .properties(height=210, title="Heatmap Vector 92 chiều sau chuẩn hóa Z-score")
    )
    return chart


def build_before_after_zscore_chart(
    vec_raw: np.ndarray,
    vec_zscore: np.ndarray,
    feature_names: list[str],
    selected_group: str,
) -> alt.Chart:
    """Side-by-side comparison of raw vs Z-score feature values within a selected group."""
    rows = []
    for fname, r_val, z_val in zip(feature_names, vec_raw, vec_zscore):
        if get_feature_group(fname) == selected_group:
            rows.append({"feature": fname, "Loại": "Trước Z-score (Raw)", "Giá trị": float(r_val)})
            rows.append({"feature": fname, "Loại": "Sau Z-score", "Giá trị": float(z_val)})
    df = pd.DataFrame(rows)

    raw_df = df[df["Loại"] == "Trước Z-score (Raw)"]
    z_df = df[df["Loại"] == "Sau Z-score"]

    c_raw = (
        alt.Chart(raw_df)
        .mark_bar(color="#64748B")
        .encode(
            x=alt.X("feature:N", sort=None, title="Đặc trưng"),
            y=alt.Y("Giá trị:Q", title="Giá trị gốc (Raw)"),
            tooltip=["feature:N", alt.Tooltip("Giá trị:Q", format=".4f")],
        )
        .properties(height=230, title=f"Trước chuẩn hóa (Raw) — {selected_group}")
    )
    c_z = (
        alt.Chart(z_df)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", sort=None, title="Đặc trưng"),
            y=alt.Y("Giá trị:Q", title="Giá trị Z-score (μ=0, σ=1)"),
            color=alt.condition(
                alt.datum["Giá trị"] >= 0,
                alt.value("#2563EB"),
                alt.value("#DC2626"),
            ),
            tooltip=["feature:N", alt.Tooltip("Giá trị:Q", format=".4f")],
        )
        .properties(height=230, title=f"Sau chuẩn hóa Z-score — {selected_group}")
    )
    return alt.hconcat(c_raw, c_z).resolve_scale(y="independent")


def build_cluster_species_bar(species_dist: dict[str, int], cluster_id: int) -> alt.Chart:
    """Horizontal bar chart showing species distribution inside a K-Means cluster."""
    df = pd.DataFrame(
        [{"Loài": sp, "Số file": cnt} for sp, cnt in species_dist.items()]
    )
    if df.empty:
        df = pd.DataFrame([{"Loài": "Trống", "Số file": 0}])
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color="#059669")
        .encode(
            x=alt.X("Số file:Q", title="Số lượng tệp WAV"),
            y=alt.Y("Loài:N", sort="-x", title="Loài động vật"),
            tooltip=["Loài:N", "Số file:Q"],
        )
        .properties(
            height=max(160, 26 * len(df)),
            title=f"Phân bố loài trong Cụm #{cluster_id}",
        )
    )
    return chart


def render_knn_results_panel(
    result: KNNSearchResult,
    accent: str = "blue",
    show_audio_players: bool = True,
) -> None:
    """Render full KNN search metadata (Section 9.4) and Top-K match cards with audio players."""
    card_cls = "match-card-km" if accent == "green" else "match-card"

    st.markdown(f"#### {result.mode_name}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Thời gian KNN", format_ms(result.search_time_ms))
    m2.metric("Số file quét", f"{result.candidates_scanned} / 648")
    m3.metric("Dự đoán loài (Weighted)", result.predicted_species_weighted)

    st.caption(
        f"Thuật toán: `{result.algorithm}` | Metric: `{result.metric}` | "
        f"K láng giềng: `{result.n_neighbors}` | Majority vote: **{result.predicted_species_majority}**"
    )
    if result.fallback_used:
        st.warning(
            f"Đã kích hoạt Fallback (Mục 8.3): cụm gốc có ít hơn {result.n_neighbors} file nên hệ thống đã mở rộng sang các cụm lân cận {result.clusters_searched}."
        )

    for match in result.matches:
        st.markdown(
            f"""
            <div class="{card_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <strong>#{match.rank} &nbsp; {match.name}</strong>
                    <div>
                        <span class="badge badge-species">Loài: {match.species}</span>
                        <span class="badge badge-cluster">Cụm #{match.cluster_id}</span>
                        <span class="badge badge-dist">d = {match.distance:.4f}</span>
                    </div>
                </div>
                <div style="font-size:0.82rem; color:#475569; margin-top:4px;">
                    Thời lượng: {match.duration:.3f}s &nbsp;|&nbsp; SR gốc: {match.frame_rate:,} Hz &nbsp;|&nbsp; Dung lượng: {format_bytes(match.size)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if show_audio_players:
            if match.audio_path is not None and Path(match.audio_path).is_file():
                st.audio(str(match.audio_path))
            else:
                st.caption(f"_Không tìm thấy file âm thanh gốc để phát (`{match.csv_path}`)._")

    # Summary dataframe for quick copy / inspection
    with st.expander("Xem bảng chi tiết Top-K và trọng số vote loài", expanded=False):
        df_m = pd.DataFrame([
            {
                "Hạng": m.rank,
                "Tên tệp": m.name,
                "Loài": m.species,
                "Cụm K-Means": m.cluster_id,
                f"Khoảng cách ({result.metric})": round(m.distance, 6),
                "Thời lượng (s)": m.duration,
                "SR gốc (Hz)": m.frame_rate,
            }
            for m in result.matches
        ])
        st.dataframe(df_m, width="stretch", hide_index=True)

        vote_df = pd.DataFrame([
            {"Loài": sp, "Trọng số (1/d chuẩn hóa)": f"{w * 100:.1f}%"}
            for sp, w in result.species_votes_weighted.items()
        ])
        st.dataframe(vote_df, width="stretch", hide_index=True)
