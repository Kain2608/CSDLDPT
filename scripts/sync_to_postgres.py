"""Đồng bộ lược đồ OOP + pgvector + K-Means và nạp toàn bộ dữ liệu lên PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from web_app.config import (
    DATA_DIR,
    DELTA_WIDTH,
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
    WIN_LENGTH,
)

SCHEMA_SQL_PATH = PROJECT_ROOT / "scripts" / "schema_oop_pgvector.sql"


def get_connection(register_vec: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5433")),
        dbname=os.getenv("PG_DBNAME", "animal_audio_oop_db"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
    )
    if register_vec:
        register_vector(conn)
    return conn


def _is_unified_schema_ready(conn: psycopg.Connection) -> bool:
    """Kiểm tra DB đã có đầy đủ lược đồ hợp nhất OOP (multimedia_entity + extraction_config + stored functions) chưa."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'multimedia_entity')
                AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'animal_audio')
                AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'kmeans_clusters')
                AND EXISTS (SELECT 1 FROM pg_inherits i
                            JOIN pg_class c ON c.oid = i.inhrelid
                            JOIN pg_class p ON p.oid = i.inhparent
                            WHERE c.relname = 'animal_audio' AND p.relname = 'multimedia_entity')
                AND EXISTS (SELECT 1 FROM pg_type WHERE typname = 'extraction_config')
                AND EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'search_similar_animals')
                AND EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'predict_nearest_clusters');
            """
        )
        return bool(cur.fetchone()[0])


def apply_unified_schema(conn: psycopg.Connection) -> None:
    """Thực thi file SQL tạo lược đồ hợp nhất OOP + pgvector + K-Means."""
    sql_text = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()
    print(f"[sync_to_postgres] Đã áp dụng lược đồ hợp nhất từ {SCHEMA_SQL_PATH.name}")


def sync_all(recreate_schema: bool = False) -> dict[str, int]:
    """Khởi tạo/đồng bộ lược đồ OOP + pgvector và nạp toàn bộ 648 file + 13 cụm + scaler."""
    df_raw = pd.read_csv(DATA_DIR / "features_raw_92d.csv")
    df_z = pd.read_csv(DATA_DIR / "features_zscore_92d.csv")
    with open(DATA_DIR / "zscore_scaler_params.json", "r", encoding="utf-8") as f:
        scaler_params = json.load(f)
    km_payload = joblib.load(MODELS_DIR / "kmeans_model.joblib")

    feature_cols = scaler_params["feature_names"]
    assert len(feature_cols) == 92, f"Số chiều đặc trưng phải bằng 92, nhận được {len(feature_cols)}!"

    km_model = km_payload["model"]
    cluster_labels = km_payload["cluster_labels"]

    # Bước 0: Kiểm tra & áp dụng lược đồ hợp nhất nếu cần (không register_vector trước khi CREATE EXTENSION)
    with get_connection(register_vec=False) as conn_init:
        if recreate_schema or not _is_unified_schema_ready(conn_init):
            apply_unified_schema(conn_init)

    # Bước 1–3: Nạp dữ liệu trong một giao dịch nguyên tử (Atomic Transaction)
    with get_connection(register_vec=True) as conn:
        with conn.cursor() as cur:
            # 1. Nạp tham số chuẩn hóa Z-score đóng băng
            cur.execute(
                """
                INSERT INTO zscore_scaler_model (model_name, feature_names, mean_vector, std_vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_name) DO UPDATE
                SET feature_names = EXCLUDED.feature_names,
                    mean_vector   = EXCLUDED.mean_vector,
                    std_vector    = EXCLUDED.std_vector,
                    updated_at    = CURRENT_TIMESTAMP;
                """,
                ("dataset_648_v1", feature_cols, scaler_params["mean"], scaler_params["std"]),
            )

            # 2. Nạp 13 tâm cụm K-Means vào kmeans_clusters
            df_z_temp = df_z.copy()
            df_z_temp["cluster_id"] = cluster_labels
            for c_id in range(km_model.n_clusters):
                sub = df_z_temp[df_z_temp["cluster_id"] == c_id]
                dist = {str(k): int(v) for k, v in sub["species"].value_counts().items()}
                dom_species = next(iter(dist.keys()), None)
                cur.execute(
                    """
                    INSERT INTO kmeans_clusters (
                        cluster_id, centroid, cluster_size, dominant_species, species_distribution
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (cluster_id) DO UPDATE
                    SET centroid             = EXCLUDED.centroid,
                        cluster_size         = EXCLUDED.cluster_size,
                        dominant_species     = EXCLUDED.dominant_species,
                        species_distribution = EXCLUDED.species_distribution,
                        updated_at           = CURRENT_TIMESTAMP;
                    """,
                    (
                        int(c_id),
                        km_model.cluster_centers_[c_id].tolist(),
                        int(len(sub)),
                        dom_species,
                        json.dumps(dist, ensure_ascii=False),
                    ),
                )

            # 3. Nạp 648 đối tượng lớp con animal_audio (kế thừa từ multimedia_entity)
            insert_sql = """
                INSERT INTO animal_audio (
                    file_name, file_path, duration_sec, species,
                    tech_specs, proc_config, features_raw, features_zscore, cluster_id
                )
                VALUES (
                    %s, %s, %s, %s,
                    ROW(%s, %s, %s, %s, %s)::wav_tech_specs,
                    ROW(%s, %s, %s, %s, %s, %s, %s)::extraction_config,
                    %s, %s, %s
                )
                ON CONFLICT (file_name) DO UPDATE
                SET file_path       = EXCLUDED.file_path,
                    duration_sec    = EXCLUDED.duration_sec,
                    species         = EXCLUDED.species,
                    tech_specs      = EXCLUDED.tech_specs,
                    proc_config     = EXCLUDED.proc_config,
                    features_raw    = EXCLUDED.features_raw,
                    features_zscore = EXCLUDED.features_zscore,
                    cluster_id      = EXCLUDED.cluster_id;
            """

            for idx in range(len(df_raw)):
                r = df_raw.iloc[idx]
                z = df_z.iloc[idx]
                vec_raw = [float(r[c]) for c in feature_cols]
                vec_z = [float(z[c]) for c in feature_cols]
                cur.execute(
                    insert_sql,
                    (
                        str(r["name"]),
                        str(r["path"]),
                        float(r["duration"]),
                        str(r["species"]),
                        int(r["channels"]),
                        int(r["sample_width"]),
                        int(r["frame_rate"]),
                        int(r["nframes"]),
                        int(r["size"]),
                        SAMPLE_RATE,
                        1,
                        WIN_LENGTH,
                        HOP_LENGTH,
                        N_FFT,
                        N_MFCC,
                        DELTA_WIDTH,
                        vec_raw,
                        vec_z,
                        int(cluster_labels[idx]),
                    ),
                )

            # 4. Kiểm chứng tính kế thừa OOP & số lượng bản ghi ngay sau khi nạp
            cur.execute("SELECT COUNT(*) FROM multimedia_entity;")
            n_parent = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM animal_audio;")
            n_child = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM kmeans_clusters;")
            n_clusters = int(cur.fetchone()[0])

        conn.commit()

    summary = {
        "multimedia_entity_count": n_parent,
        "animal_audio_count": n_child,
        "kmeans_clusters_count": n_clusters,
    }
    print(
        f"[sync_to_postgres] Thành công! multimedia_entity={n_parent} (kế thừa OOP), "
        f"animal_audio={n_child}, kmeans_clusters={n_clusters}, scaler=dataset_648_v1"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ lược đồ OOP + pgvector và nạp dữ liệu lên PostgreSQL.")
    parser.add_argument(
        "--recreate-schema",
        action="store_true",
        help="Tạo mới lại toàn bộ lược đồ OOP + pgvector từ scripts/schema_oop_pgvector.sql trước khi nạp.",
    )
    args = parser.parse_args()
    sync_all(recreate_schema=args.recreate_schema)
