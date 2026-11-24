"""End-to-end tests for the complete AnimalSoundPipeline."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from web_app.config import resolve_audio_path
from web_app.engine.pipeline import AnimalSoundPipeline


def test_pipeline_end_to_end_with_real_wav_bytes() -> None:
    pipeline = AnimalSoundPipeline.load()
    sample_row = pipeline.df_zscore.iloc[0]
    audio_path = resolve_audio_path(str(sample_row["path"]))

    if audio_path is not None:
        raw_bytes = audio_path.read_bytes()
        fname = audio_path.name
    else:
        t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, 0.3 * np.sin(2 * np.pi * 500 * t), 16000, format="WAV")
        raw_bytes = buf.getvalue()
        fname = "synthetic.wav"

    res = pipeline.query_audio_bytes(
        audio_bytes=raw_bytes,
        filename=fname,
        mode="compare",
        top_k=5,
        include_advanced_strategies=True,
    )

    assert res.extraction.vector_raw.shape == (92,)
    assert res.scaling.vector_zscore.shape == (92,)
    assert res.result_without_kmeans is not None
    assert res.result_with_kmeans is not None
    assert res.result_multi_cluster is not None
    assert res.result_pca_kmeans is not None
    assert res.comparison_summary is not None
    assert res.total_end_to_end_ms > 0.0
    assert len(res.result_without_kmeans.matches) == 5
    assert len(res.result_with_kmeans.matches) == 5

    # When querying an exact dataset WAV file, Top-1 in both engines should match itself
    if audio_path is not None:
        assert res.result_without_kmeans.matches[0].name == sample_row["name"]
        assert res.result_without_kmeans.matches[0].distance < 1e-2
        assert res.result_with_kmeans.matches[0].name == sample_row["name"]
        assert res.result_with_kmeans.matches[0].distance < 1e-2


def test_pipeline_query_dataset_sample() -> None:
    pipeline = AnimalSoundPipeline.load()
    res = pipeline.query_dataset_sample(row_index=100, mode="compare", top_k=5)
    assert res.comparison_summary is not None
    assert res.comparison_summary["full_candidates"] == 648
    assert res.comparison_summary["kmeans_candidates"] < 648
