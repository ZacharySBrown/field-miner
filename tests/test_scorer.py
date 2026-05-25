"""Tests for the Scorer module."""

import pytest

from field_miner.scorer import Scorer
from tests.conftest import _make_window_features


def test_fast_mode_no_fit_required(default_config):
    default_config.fast = True
    scorer = Scorer(default_config, fast=True)
    wf = _make_window_features()
    sw = scorer.score(wf)
    assert 0.0 <= sw.score <= 1.0


def test_score_requires_fit(default_config):
    scorer = Scorer(default_config, fast=False)
    wf = _make_window_features()
    with pytest.raises(ValueError, match="fit"):
        scorer.score(wf)


def test_score_range_after_fit(default_config):
    scorer = Scorer(default_config, fast=False)
    windows = [
        _make_window_features(
            spectral_centroid_std=i * 100,
            mfcc_delta_std=i * 2,
            onset_count=i,
            rms_variance=i * 0.01,
            harmonic_ratio=i * 0.1,
        )
        for i in range(1, 20)
    ]
    scorer.fit(windows)
    for wf in windows:
        sw = scorer.score(wf)
        assert 0.0 <= sw.score <= 1.0
        for name, val in sw.sub_scores.items():
            assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"


def test_chirp_beats_silence(default_config):
    default_config.fast = True
    scorer = Scorer(default_config, fast=True)

    silence = _make_window_features(
        spectral_centroid_std=0, spectral_rolloff_std=0,
        mfcc_delta_std=0, onset_count=0, rms_variance=0,
        mel_entropy=0, harmonic_ratio=0,
    )
    chirp = _make_window_features(
        spectral_centroid_std=1500, spectral_rolloff_std=3000,
        mfcc_delta_std=10, onset_count=6, rms_variance=0.05,
        mel_entropy=4.0, harmonic_ratio=0.4,
    )

    sw_silence = scorer.score(silence)
    sw_chirp = scorer.score(chirp)
    assert sw_chirp.score > sw_silence.score


def test_noise_beats_silence(default_config):
    default_config.fast = True
    scorer = Scorer(default_config, fast=True)

    silence = _make_window_features(
        spectral_centroid_std=0, spectral_rolloff_std=0,
        mfcc_delta_std=0, onset_count=0, rms_variance=0,
        mel_entropy=0, harmonic_ratio=0,
    )
    noise = _make_window_features(
        spectral_centroid_std=800, spectral_rolloff_std=1500,
        mfcc_delta_std=5, onset_count=4, rms_variance=0.03,
        mel_entropy=3.5, harmonic_ratio=0.15,
    )

    sw_silence = scorer.score(silence)
    sw_noise = scorer.score(noise)
    assert sw_noise.score > sw_silence.score


def test_rescore_consistency(default_config):
    default_config.fast = True
    scorer = Scorer(default_config, fast=True)
    wf = _make_window_features()
    sw1 = scorer.score(wf)
    sw2 = scorer.score(wf)
    assert sw1.score == sw2.score


def test_calibrate_from_sample(default_config):
    scorer = Scorer(default_config)
    windows = [
        _make_window_features(spectral_centroid_std=i * 50, onset_count=i)
        for i in range(100)
    ]
    scorer.calibrate_from_sample(windows, sample_size=30)
    sw = scorer.score(windows[50])
    assert 0.0 <= sw.score <= 1.0
