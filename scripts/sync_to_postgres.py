import json
import os
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "web_app" / "data"
MODELS_DIR = PROJECT_ROOT / "web_app" / "models"


def get_connection():
    conn = psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5433")),
        dbname=os.getenv("PG_DBNAME", "animal_audio_oop_db"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
    )
    register_vector(conn)
    return conn


def sync_all():
    df_raw = pd.read_csv(DATA_DIR / "features_raw_92d.csv")
    df_z = pd.read_csv(DATA_DIR / "features_zscore_92d.csv")
    with open(DATA_DIR / "zscore_scaler_params.json", "r", encoding="utf-8") as f:
        scaler_params = json.load(f)
    km_payload = joblib.load(MODELS_DIR / "kmeans_model.joblib")

    feature_cols = scaler_params["feature_names"]
    km_model = km_payload["model"]
    cluster_labels = km_payload["cluster_labels"]

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Nạp zscore_scaler_model
            cur.execute(
                """
                INSERT INTO zscore_scaler_model (model_name, feature_names, mean_vector, std_vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_name) DO UPDATE
                SET feature_names = EXCLUDED.feature_names,
                    mean_vector   = EXCLUDED.mean_vector,
                    std_vector    = EXCLUDED.std_vector,
                    updated_at    = NOW();
                """,
                ("dataset_648_v1", feature_cols, scaler_params["mean"], scaler_params["std"]),
            )

            # 2. Nạp 13 tâm cụm vào kmeans_clusters
            df_z_temp = df_z.copy()
            df_z_temp["cluster_id"] = cluster_labels
            for c_id in range(km_model.n_clusters):
                sub = df_z_temp[df_z_temp["cluster_id"] == c_id]
                dist = sub["species"].value_counts().to_dict()
                dom_species = next(iter(dist.keys()), None)
                cur.execute(
                    """
                    INSERT INTO kmeans_clusters (cluster_id, centroid, cluster_size, dominant_species, species_distribution)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (cluster_id) DO UPDATE
                    SET centroid             = EXCLUDED.centroid,
                        cluster_size         = EXCLUDED.cluster_size,
                        dominant_species     = EXCLUDED.dominant_species,
                        species_distribution = EXCLUDED.species_distribution,
                        updated_at           = NOW();
                    """,
                    (c_id, km_model.cluster_centers_[c_id].tolist(), len(sub), dom_species, json.dumps(dist)),
                )

            # 3. Nạp 648 file âm thanh vào animal_audio
            for idx in range(len(df_raw)):
                r = df_raw.iloc[idx]
                z = df_z.iloc[idx]
                vec_raw = [float(r[c]) for c in feature_cols]
                vec_z = [float(z[c]) for c in feature_cols]
                cur.execute(
                    """
                    INSERT INTO animal_audio (
                        file_name, file_path, duration_sec, species,
                        tech_specs, features_raw, features_zscore, cluster_id
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        ROW(%s, %s, %s, %s, %s)::wav_tech_specs,
                        %s, %s, %s
                    )
                    ON CONFLICT (file_name) DO UPDATE
                    SET features_raw    = EXCLUDED.features_raw,
                        features_zscore = EXCLUDED.features_zscore,
                        cluster_id      = EXCLUDED.cluster_id;
                    """,
                    (
                        str(r["name"]), str(r["path"]), float(r["duration"]), str(r["species"]),
                        int(r["channels"]), int(r["sample_width"]), int(r["frame_rate"]),
                        int(r["nframes"]), int(r["size"]),
                        vec_raw, vec_z, int(cluster_labels[idx]),
                    ),
                )
        conn.commit()
    print(f"Đã nạp xong 1 scaler, {km_model.n_clusters} cụm K-Means và {len(df_raw)} file âm thanh lên PostgreSQL!")


if __name__ == "__main__":
    sync_all()