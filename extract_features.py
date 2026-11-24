import json
import wave
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

# Cấu hình tham số cố định theo thiết kế ở Câu 2
SAMPLE_RATE = 16000
WIN_LENGTH = 400     # 25 ms
HOP_LENGTH = 160     # 10 ms
N_FFT = 512
N_MFCC = 13
DELTA_WIDTH = 9      # K = 4

DATASET_DIR = Path("Animal-Soundprepros")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_wav_metadata(file_path: Path) -> dict:
    """Trích xuất 8 trường metadata gốc của tệp WAV."""
    with wave.open(str(file_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / float(frame_rate) if frame_rate > 0 else 0.0
    return {
        "name": file_path.name,
        "path": str(file_path),
        "species": file_path.parent.name,
        "channels": channels,
        "sample_width": sample_width,
        "frame_rate": frame_rate,
        "nframes": nframes,
        "duration": round(duration, 4),
        "size": file_path.stat().st_size,
    }


def extract_92d_features(file_path: Path) -> np.ndarray:
    """Trích rút vector đặc trưng Bộ C (92 chiều) cho một tệp âm thanh."""
    # 1. Tải tín hiệu, chuẩn hóa Mono & 16.000 Hz
    y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError(f"Tệp âm thanh rỗng: {file_path}")

    # 2. Phân khung (cửa sổ Hann) và biến đổi STFT
    stft_matrix = librosa.stft(
        y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window="hann"
    )
    S = np.abs(stft_matrix)   # Phổ biên độ
    P = S ** 2                # Phổ công suất

    # 3. Trích rút 7 đặc trưng cơ bản -> Bộ A (14 chiều: 0..13)
    rms = librosa.feature.rms(y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=N_FFT)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=N_FFT)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=N_FFT, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S, power=2.0)[0]

    # Tính Spectral Flux từ phổ biên độ đã chuẩn hóa theo từng khung
    S_norm = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-10)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1, prepend=S_norm[:, :1]) ** 2, axis=0))

    vec_A = []
    for feat in [rms, zcr, centroid, bandwidth, rolloff, flatness, flux]:
        vec_A.extend([float(np.mean(feat)), float(np.std(feat))])

    # 4. Trích rút MFCC, Delta, Delta-Delta -> Bộ B (78 chiều: 14..91)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(P), sr=sr, n_mfcc=N_MFCC)
    n_frames = mfcc.shape[1]

    # Xử lý linh hoạt cho tệp ngắn có ít hơn 9 khung
    actual_width = min(DELTA_WIDTH, n_frames if n_frames % 2 == 1 else n_frames - 1)
    actual_width = max(3, actual_width)

    delta_mfcc = librosa.feature.delta(mfcc, width=actual_width, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, width=actual_width, order=2)

    # 5. Tổng hợp thống kê và ghép nối thành Bộ C (92 chiều)
    vec_C = np.concatenate([
        np.array(vec_A, dtype=np.float64),
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(np.abs(delta_mfcc), axis=1),
        np.std(delta_mfcc, axis=1),
        np.mean(np.abs(delta2_mfcc), axis=1),
        np.std(delta2_mfcc, axis=1),
    ])

    assert vec_C.shape == (92,), f"Số chiều không hợp lệ: {vec_C.shape}"
    return np.nan_to_num(vec_C, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_columns() -> list:
    """Sinh danh sách tên 92 cột đặc trưng theo đúng bố cục Bảng 3.3."""
    basic = ["rms", "zcr", "centroid", "bandwidth", "rolloff", "flatness", "flux"]
    cols = [f"{name}_{stat}" for name in basic for stat in ("mean", "std")]
    cols += [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    cols += [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    cols += [f"delta_{i}_mean_abs" for i in range(1, N_MFCC + 1)]
    cols += [f"delta_{i}_std" for i in range(1, N_MFCC + 1)]
    cols += [f"delta2_{i}_mean_abs" for i in range(1, N_MFCC + 1)]
    cols += [f"delta2_{i}_std" for i in range(1, N_MFCC + 1)]
    return cols


def run_batch_extraction():
    wav_files = sorted(DATASET_DIR.rglob("*.wav"))
    records, feature_matrix, failed = [], [], []

    for wav_path in wav_files:
        try:
            records.append(get_wav_metadata(wav_path))
            feature_matrix.append(extract_92d_features(wav_path))
        except Exception as err:
            failed.append({"path": str(wav_path), "error": str(err)})

    X_raw = np.vstack(feature_matrix)
    cols = build_feature_columns()

    # Tính tham số và chuẩn hóa Z-score trên toàn bộ bộ dữ liệu
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0)
    stds_safe = np.where(stds == 0, 1.0, stds)
    X_zscore = (X_raw - means) / stds_safe

    df_meta = pd.DataFrame(records)
    pd.concat([df_meta, pd.DataFrame(X_raw, columns=cols)], axis=1).to_csv(
        OUTPUT_DIR / "features_raw_92d.csv", index=False
    )
    pd.concat([df_meta, pd.DataFrame(X_zscore, columns=cols)], axis=1).to_csv(
        OUTPUT_DIR / "features_zscore_92d.csv", index=False
    )

    with open(OUTPUT_DIR / "zscore_scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"feature_names": cols, "mean": means.tolist(), "std": stds_safe.tolist()},
            f, indent=2, ensure_ascii=False
        )

    if failed:
        pd.DataFrame(failed).to_csv(OUTPUT_DIR / "failed_files.csv", index=False)


if __name__ == "__main__":
    run_batch_extraction()