# Hệ Thống Tìm Kiếm Âm Thanh Động Vật Tương Đồng (K-Means + KNN)

Dự án AI tìm kiếm âm thanh động vật tương đồng dựa trên **vector đặc trưng 92 chiều (Bộ C)**, chuẩn hóa **Z-score đóng băng**, phân cụm **K-Means (K=13)** và tìm kiếm láng giềng gần nhất **KNN (KDTree)**, triển khai kèm giao diện web đa trang bằng **Streamlit**.

## 1. Cấu trúc dự án

```text
Animal_Sound/
├── .venv/                         # Môi trường ảo Python 3.11 độc lập
├── .streamlit/config.toml         # Cấu hình giao diện & server Streamlit
├── pyproject.toml                 # Metadata dự án & phụ thuộc (PEP 621)
├── requirements.txt               # Danh sách thư viện cài đặt qua pip
├── scripts/
│   ├── prepare_data.py            # Copy & kiểm tra tính toàn vẹn dữ liệu 92D vào web_app/data/
│   ├── train_models.py            # Huấn luyện offline KMeans (K=13), fit KDTree, PCA/t-SNE
│   └── run_benchmark.py           # Benchmark K=2..20, 200 queries, xuất biểu đồ báo cáo PNG
├── web_app/
│   ├── config.py                  # Cấu hình tập trung (DSP, hyperparameters, đường dẫn)
│   ├── ui_components.py           # Thành phần giao diện và biểu đồ Altair tái sử dụng
│   ├── app.py                     # Trang chủ (Tổng quan & Kiến trúc Pipeline)
│   ├── pages/
│   │   ├── 1_search.py            # Trang Tìm kiếm & So sánh 2 bản (Có / Không K-Means)
│   │   ├── 2_report.py            # Trang Báo cáo Kỹ thuật & Benchmark
│   │   └── 3_clusters.py          # Trang Trực quan hóa Cụm (PCA 2D / t-SNE 2D / Heatmap)
│   ├── engine/
│   │   ├── feature_engine.py      # Trích xuất 92D (SR=16kHz, win=400, hop=160, n_fft=512)
│   │   ├── scaler.py              # Chuẩn hóa Z-score bằng μ, σ từ zscore_scaler_params.json
│   │   ├── kmeans_engine.py       # Huấn luyện, dự đoán cụm & đánh giá Silhouette/ARI/NMI/DB
│   │   ├── knn_engine.py          # Global KDTree, Cluster KNN (+Fallback), Top-2 cụm, PCA+KM+KNN
│   │   └── pipeline.py            # Điều phối truy vấn end-to-end & đo thời gian từng bước
│   ├── data/                      # Dữ liệu CSV 92D & zscore_scaler_params.json
│   ├── models/                    # Mô hình .joblib, model_metadata.json, projections_2d.csv
│   └── static/                    # Ảnh PNG báo cáo (elbow_silhouette.png, pca_scatter.png, ...)
└── tests/                         # Bộ kiểm thử tự động pytest (unit + end-to-end)
```

## 2. Cài đặt môi trường ảo (`.venv`)

```powershell
# 1. Tạo môi trường ảo Python 3.11
python -m venv .venv

# 2. Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

## 3. Huấn luyện mô hình & Chạy Benchmark offline

```powershell
# Bước 1: Copy và kiểm tra dữ liệu 92D từ output/ sang web_app/data/
python scripts/prepare_data.py

# Bước 2: Huấn luyện K-Means (K=13), fit KDTree và tạo tọa độ PCA/t-SNE
python scripts/train_models.py

# Bước 3: Chạy benchmark (K=2..20, 200 queries, 4 chiến lược) và xuất ảnh PNG báo cáo
python scripts/run_benchmark.py
```

## 4. Chạy ứng dụng Web Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run web_app/app.py
```

*(Tùy chọn: đặt biến môi trường `ANIMAL_SOUND_AUDIO_DIR` trỏ tới thư mục chứa các tệp WAV gốc `Animal-Soundprepros` để phát trực tiếp âm thanh Top-5 trong giao diện web.)*

## 5. Chạy bộ kiểm thử tự động (`pytest`)

```powershell
.\.venv\Scripts\pytest.exe
```
