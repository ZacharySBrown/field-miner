"""Pytest fixtures with synthetic audio for testing."""

from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import soundfile as sf

from field_miner.models import Config, WindowFeatures


@pytest.fixture
def silence_array():
    return np.zeros(44100 * 20, dtype=np.float32), 44100


@pytest.fixture
def clipped_array():
    t = np.linspace(0, 20, 44100 * 20, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t) * 1.5  # way over 1.0
    return np.clip(y, -1.01, 1.01).astype(np.float32), 44100


@pytest.fixture
def noise_array():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(44100 * 20).astype(np.float32) * 0.3
    return y, 44100


@pytest.fixture
def chirp_array():
    t = np.linspace(0, 20, 44100 * 20)
    y = scipy.signal.chirp(t, f0=200, f1=4000, t1=20, method="logarithmic")
    return y.astype(np.float32) * 0.5, 44100


@pytest.fixture
def default_config(tmp_path):
    return Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / ".cache"),
    )


@pytest.fixture
def test_wav(tmp_path):
    """Generate a 20s WAV with pink noise + sine bursts."""
    rng = np.random.default_rng(123)
    sr = 44100
    duration = 20
    n_samples = sr * duration

    noise = rng.standard_normal(n_samples).astype(np.float32) * 0.05

    # Add sine bursts at 3s and 12s
    for t_start in [3.0, 12.0]:
        start = int(t_start * sr)
        end = int((t_start + 1.5) * sr)
        freq = 880
        t = np.linspace(0, 1.5, end - start)
        noise[start:end] += (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    out_path = tmp_path / "test_20s.wav"
    sf.write(str(out_path), noise, sr)
    return out_path


def _make_window_features(**overrides) -> WindowFeatures:
    """Helper to create WindowFeatures with sensible defaults."""
    defaults = dict(
        source_file="test.wav",
        start_s=0.0,
        end_s=20.0,
        peak_amplitude=0.5,
        rms_mean_db=-20.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
        spectral_centroid_mean=2000.0,
        spectral_centroid_std=500.0,
        spectral_rolloff_mean=4000.0,
        spectral_rolloff_std=1000.0,
        spectral_contrast=[20.0] * 7,
        spectral_flatness_mean=0.3,
        spectral_flatness_std=0.1,
        mfcc_mean=[0.0] * 13,
        mfcc_std=[1.0] * 13,
        mfcc_delta_std=5.0,
        zcr_mean=0.05,
        zcr_std=0.02,
        onset_strength_mean=3.0,
        onset_strength_std=1.0,
        onset_count=8,
        rms_variance=0.02,
        mel_entropy=3.0,
        harmonic_ratio=0.4,
        top_yamnet_labels=[],
        yamnet_embedding=[],
    )
    defaults.update(overrides)
    return WindowFeatures(**defaults)
