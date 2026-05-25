"""Tests for the Filter module."""

from field_miner.filter import Filter
from field_miner.models import Config
from tests.conftest import _make_window_features


def test_hard_reject_silence(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(rms_mean_db=-65)
    rejected, reason = filt.hard_reject(wf)
    assert rejected is True
    assert "silence" in reason


def test_hard_reject_clipping(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(peak_amplitude=1.0, clipping_ratio=0.01)
    rejected, reason = filt.hard_reject(wf)
    assert rejected is True
    assert "clipping" in reason


def test_hard_reject_dc_offset(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(dc_offset=0.1)
    rejected, reason = filt.hard_reject(wf)
    assert rejected is True
    assert "dc_offset" in reason


def test_hard_reject_speech_label(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(
        top_yamnet_labels=[("Speech", 0.9), ("Music", 0.05), ("Noise", 0.02)]
    )
    rejected, reason = filt.hard_reject(wf)
    assert rejected is True
    assert "Speech" in reason


def test_no_reject_normal(default_config):
    filt = Filter(default_config)
    wf = _make_window_features()
    rejected, reason = filt.hard_reject(wf)
    assert rejected is False
    assert reason is None


def test_prefer_boost_birds(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(
        top_yamnet_labels=[("Bird", 0.8), ("Water", 0.1), ("Wind", 0.05)]
    )
    boost = filt.compute_prefer_boost(wf)
    assert boost > 1.0


def test_wind_penalty(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(
        top_yamnet_labels=[("Bird", 0.5), ("Wind noise", 0.3)]
    )
    score = filt.apply_soft_penalties(wf, 0.8)
    assert score < 0.8


def test_boring_drone_penalty(default_config):
    filt = Filter(default_config)
    wf = _make_window_features(onset_count=0, spectral_centroid_std=100)
    score = filt.apply_soft_penalties(wf, 0.8)
    assert score < 0.8


def test_rejection_summary(default_config):
    filt = Filter(default_config)
    filt.hard_reject(_make_window_features(rms_mean_db=-65))
    filt.hard_reject(_make_window_features(rms_mean_db=-65))
    filt.hard_reject(_make_window_features(peak_amplitude=0.5, clipping_ratio=0.01))
    summary = filt.rejection_summary()
    assert summary["silence"] == 2
    assert summary["clipping_ratio"] == 1
