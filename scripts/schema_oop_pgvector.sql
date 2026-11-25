-- ============================================================================
-- Lược đồ CSDL Hợp nhất: Hướng Đối tượng – Quan hệ (OOP) + pgvector + K-Means
-- Database: animal_audio_oop_db (PostgreSQL 17 + pgvector >= 0.7)
-- Đồng bộ hoàn toàn giữa Báo cáo Mục 3.2 (OOP) và Mục 4 (K-Means + KNN)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Dọn sạch lược đồ cũ khi khởi tạo lại từ đầu (theo đúng thứ tự phụ thuộc)
DROP FUNCTION IF EXISTS search_similar_animals(vector, CHAR, VARCHAR, INT[], INT);
DROP FUNCTION IF EXISTS search_similar_animals(REAL[], CHAR, INT);
DROP FUNCTION IF EXISTS predict_nearest_clusters(vector);
DROP FUNCTION IF EXISTS cosine_similarity(REAL[], REAL[]);

DROP TABLE IF EXISTS animal_audio CASCADE;
DROP TABLE IF EXISTS multimedia_entity CASCADE;
DROP TABLE IF EXISTS kmeans_clusters CASCADE;
DROP TABLE IF EXISTS zscore_scaler_model CASCADE;

DROP TYPE IF EXISTS wav_tech_specs CASCADE;
DROP TYPE IF EXISTS extraction_config CASCADE;

-- ============================================================================
-- 1. CÁC KIỂU DỮ LIỆU PHỨC HỢP DO NGƯỜI DÙNG ĐỊNH NGHĨA (UDT - Composite Types)
-- ============================================================================

CREATE TYPE wav_tech_specs AS (
    channels     INT,
    sample_width INT,
    frame_rate   INT,
    nframes      BIGINT,
    size_bytes   BIGINT
);

CREATE TYPE extraction_config AS (
    target_sample_rate INT,
    target_channels    INT,
    win_length         INT,
    hop_length         INT,
    n_fft              INT,
    n_mfcc             INT,
    delta_width        INT
);

-- ============================================================================
-- 2. LỚP QUẢN TRỊ THAM SỐ CHUẨN HÓA Z-SCORE (μ, σ ĐÓNG BĂNG)
-- ============================================================================

CREATE TABLE zscore_scaler_model (
    model_id      SERIAL PRIMARY KEY,
    model_name    VARCHAR(100) UNIQUE NOT NULL,
    feature_names TEXT[] NOT NULL,
    mean_vector   DOUBLE PRECISION[] NOT NULL,
    std_vector    DOUBLE PRECISION[] NOT NULL,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_scaler_feature_dim CHECK (array_length(feature_names, 1) = 92),
    CONSTRAINT chk_scaler_mean_dim    CHECK (array_length(mean_vector, 1) = 92),
    CONSTRAINT chk_scaler_std_dim     CHECK (array_length(std_vector, 1) = 92)
);

-- ============================================================================
-- 3. LỚP PHÂN CỤM K-MEANS (K = 13 TÂM CỤM 92 CHIỀU)
-- ============================================================================

CREATE TABLE kmeans_clusters (
    cluster_id           INT PRIMARY KEY,
    centroid             vector(92) NOT NULL,
    cluster_size         INT NOT NULL,
    dominant_species     VARCHAR(100),
    species_distribution JSONB,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 4. PHÂN CẤP LỚP KẾ THỪA OOP (Superclass -> Subclass qua INHERITS)
-- ============================================================================

CREATE TABLE multimedia_entity (
    oid          SERIAL PRIMARY KEY,
    file_name    VARCHAR(255) NOT NULL,
    file_path    TEXT NOT NULL UNIQUE,
    duration_sec REAL NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE animal_audio (
    species         VARCHAR(100) NOT NULL,
    tech_specs      wav_tech_specs NOT NULL,
    proc_config     extraction_config NOT NULL
                    DEFAULT ROW(16000, 1, 400, 160, 512, 13, 9)::extraction_config,
    features_raw    vector(92) NOT NULL,
    features_zscore vector(92) NOT NULL,
    cluster_id      INT REFERENCES kmeans_clusters(cluster_id),
    -- Khai báo lại PK/UNIQUE trên lớp con (do PostgreSQL INHERITS không tự kế thừa PK/UNIQUE)
    PRIMARY KEY (oid),
    UNIQUE (file_name),
    UNIQUE (file_path)
) INHERITS (multimedia_entity);

-- Các chỉ mục tăng tốc lọc theo loài, lọc cụm K-Means và tìm kiếm vector HNSW
CREATE INDEX idx_animal_audio_species
    ON animal_audio (species);

CREATE INDEX idx_animal_audio_cluster_id
    ON animal_audio (cluster_id);

CREATE INDEX idx_animal_audio_zscore_hnsw_l2
    ON animal_audio USING hnsw (features_zscore vector_l2_ops);

CREATE INDEX idx_animal_audio_zscore_hnsw_cos
    ON animal_audio USING hnsw (features_zscore vector_cosine_ops);

-- ============================================================================
-- 5. ĐÓNG GÓI PHƯƠNG THỨC TRONG CSDL (Stored Methods / Message Passing)
-- ============================================================================

-- Phương thức 1: Định tuyến cụm K-Means gần nhất cho vector truy vấn 92D
CREATE OR REPLACE FUNCTION predict_nearest_clusters(
    query_zscore vector(92)
)
RETURNS TABLE (
    cluster_id           INT,
    centroid             vector(92),
    cluster_size         INT,
    dominant_species     VARCHAR(100),
    species_distribution JSONB,
    distance             DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        k.cluster_id,
        k.centroid,
        k.cluster_size,
        k.dominant_species,
        k.species_distribution,
        (k.centroid <-> query_zscore)::DOUBLE PRECISION AS distance
    FROM kmeans_clusters k
    ORDER BY k.centroid <-> query_zscore ASC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Phương thức 2: Tìm Top-K âm thanh tương đồng (hỗ trợ Bộ A/B/C, Euclidean/Cosine, Full-scan/K-Means)
CREATE OR REPLACE FUNCTION search_similar_animals(
    query_zscore    vector(92),
    feature_set     CHAR(1)     DEFAULT 'C',
    metric_name     VARCHAR(20) DEFAULT 'euclidean',
    target_clusters INT[]       DEFAULT NULL,
    top_k           INT         DEFAULT 5
)
RETURNS TABLE (
    rank_no          BIGINT,
    oid              INT,
    file_name        VARCHAR(255),
    species          VARCHAR(100),
    file_path        TEXT,
    cluster_id       INT,
    duration_sec     REAL,
    frame_rate       INT,
    size_bytes       BIGINT,
    distance         DOUBLE PRECISION,
    similarity_score DOUBLE PRECISION
) AS $$
DECLARE
    fs TEXT := UPPER(TRIM(COALESCE(feature_set, 'C')));
    mt TEXT := LOWER(TRIM(COALESCE(metric_name, 'euclidean')));
BEGIN
    IF fs = 'A' THEN
        -- Bộ A: 14 chiều đặc trưng cơ bản (vị trí 1..14)
        IF mt = 'cosine' THEN
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 - t.d)::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (subvector(a.features_zscore, 1, 14) <=> subvector(query_zscore, 1, 14))::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY d ASC
                LIMIT top_k
            ) t;
        ELSE
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 / (1.0 + t.d))::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (subvector(a.features_zscore, 1, 14) <-> subvector(query_zscore, 1, 14))::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY d ASC
                LIMIT top_k
            ) t;
        END IF;

    ELSIF fs = 'B' THEN
        -- Bộ B: 78 chiều MFCC + Delta + Delta-Delta (vị trí 15..92)
        IF mt = 'cosine' THEN
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 - t.d)::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (subvector(a.features_zscore, 15, 78) <=> subvector(query_zscore, 15, 78))::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY d ASC
                LIMIT top_k
            ) t;
        ELSE
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 / (1.0 + t.d))::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (subvector(a.features_zscore, 15, 78) <-> subvector(query_zscore, 15, 78))::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY d ASC
                LIMIT top_k
            ) t;
        END IF;

    ELSE
        -- Bộ C (Mặc định): Toàn bộ 92 chiều (tận dụng trực tiếp chỉ mục HNSW trên features_zscore)
        IF mt = 'cosine' THEN
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 - t.d)::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (a.features_zscore <=> query_zscore)::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY a.features_zscore <=> query_zscore ASC
                LIMIT top_k
            ) t;
        ELSE
            RETURN QUERY
            SELECT
                ROW_NUMBER() OVER (ORDER BY d ASC),
                t.oid, t.file_name, t.species, t.file_path, t.cluster_id,
                t.duration_sec, t.frame_rate, t.size_bytes,
                t.d AS distance,
                (1.0 / (1.0 + t.d))::DOUBLE PRECISION AS similarity_score
            FROM (
                SELECT a.oid, a.file_name, a.species, a.file_path, a.cluster_id,
                       a.duration_sec, (a.tech_specs).frame_rate, (a.tech_specs).size_bytes,
                       (a.features_zscore <-> query_zscore)::DOUBLE PRECISION AS d
                FROM animal_audio a
                WHERE target_clusters IS NULL OR a.cluster_id = ANY(target_clusters)
                ORDER BY a.features_zscore <-> query_zscore ASC
                LIMIT top_k
            ) t;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;
