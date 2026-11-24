import json
import os
from pathlib import Path
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("PG_DBNAME", "animal_audio_oop_db"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5433"),
}

OUTPUT_DIR = Path("output")


def import_data():
    # 1. Đọc 3 tệp dữ liệu đã trích xuất ở Mục 3.1
    df_raw = pd.read_csv(OUTPUT_DIR / "features_raw_92d.csv")
    df_zscore = pd.read_csv(OUTPUT_DIR / "features_zscore_92d.csv")
    with open(
        OUTPUT_DIR / "zscore_scaler_params.json", "r", encoding="utf-8"
    ) as f:
        scaler_params = json.load(f)

    feature_cols = scaler_params["feature_names"]
    assert len(feature_cols) == 92, "Số chiều đặc trưng phải bằng 92!"

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # 2. Nạp tham số Z-score (μ, σ) vào bảng zscore_scaler_model
        cur.execute(
            """
            INSERT INTO zscore_scaler_model (model_name, feature_names, mean_vector, std_vector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_name) DO UPDATE 
            SET mean_vector = EXCLUDED.mean_vector, std_vector = EXCLUDED.std_vector;
        """,
            (
                "dataset_648_v1",
                feature_cols,
                scaler_params["mean"],
                scaler_params["std"],
            ),
        )

        # 3. Nạp từng đối tượng âm thanh vào lớp con animal_audio
        insert_sql = """
            INSERT INTO animal_audio (
                file_name, file_path, duration_sec, species, 
                tech_specs, features_raw, features_zscore
            )
            VALUES (
                %s, %s, %s, %s, 
                ROW(%s, %s, %s, %s, %s)::wav_tech_specs, 
                %s::real[], %s::real[]
            );
        """

        for idx in range(len(df_raw)):
            row_r = df_raw.iloc[idx]
            row_z = df_zscore.iloc[idx]

            vec_raw = [float(row_r[c]) for c in feature_cols]
            vec_zscore = [float(row_z[c]) for c in feature_cols]

            cur.execute(
                insert_sql,
                (
                    row_r["name"],
                    row_r["path"],
                    float(row_r["duration"]),
                    row_r["species"],
                    int(row_r["channels"]),
                    int(row_r["sample_width"]),
                    int(row_r["frame_rate"]),
                    int(row_r["nframes"]),
                    int(row_r["size"]),
                    vec_raw,
                    vec_zscore,
                ),
            )

        conn.commit()
        print(
            f"Đã nạp thành công {len(df_raw)} đối tượng âm thanh và tham số Z-score vào PostgreSQL!"
        )
    except Exception as e:
        conn.rollback()
        print("Lỗi khi nạp dữ liệu:", e)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import_data()