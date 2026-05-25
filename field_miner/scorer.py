"""Composite interestingness scoring for audio windows."""

from __future__ import annotations

import math
import random

import numpy as np

from field_miner.models import Config, ScoredWindow, WindowFeatures


class Scorer:
    def __init__(self, config: Config, fast: bool = False):
        self.config = config
        self.fast = fast
        self._fitted = False
        self._percentiles: dict[str, tuple[float, float]] = {}  # name -> (p5, p95)

    # --- Raw sub-score computations ---

    @staticmethod
    def _spectral_dynamism(f: WindowFeatures) -> float:
        return (f.spectral_centroid_std / 1000) + (f.spectral_rolloff_std / 2000)

    @staticmethod
    def _textural_richness(f: WindowFeatures) -> float:
        contrast_mean = float(np.mean(f.spectral_contrast)) if f.spectral_contrast else 0.0
        return (contrast_mean / 40) + (f.mel_entropy / 5)

    @staticmethod
    def _timbral_movement(f: WindowFeatures) -> float:
        return f.mfcc_delta_std

    @staticmethod
    def _event_interest(f: WindowFeatures) -> float:
        # Wider sweet spot: peaks at 8, but stays high up to ~40 onsets
        # Field recordings (fire, water) often have 20-100+ micro-onsets
        return math.exp(-0.5 * ((f.onset_count - 8) / 15) ** 2)

    @staticmethod
    def _dynamic_range(f: WindowFeatures) -> float:
        return f.rms_variance

    @staticmethod
    def _harmonic_content(f: WindowFeatures) -> float:
        return max(0.0, 1 - abs(f.harmonic_ratio - 0.4) / 0.6)

    _SUB_SCORE_FUNCS = {
        "spectral_dynamism": _spectral_dynamism,
        "textural_richness": _textural_richness,
        "timbral_movement": _timbral_movement,
        "event_interest": _event_interest,
        "dynamic_range": _dynamic_range,
        "harmonic_content": _harmonic_content,
    }

    # Hand-tuned divisors for fast mode (skip percentile normalization)
    # Calibrated against real field recordings (fire, ambience)
    _FAST_DIVISORS = {
        "spectral_dynamism": 3.0,
        "textural_richness": 1.5,
        "timbral_movement": 3.0,
        "event_interest": 1.0,  # already 0-1 from gaussian
        "dynamic_range": 0.005,
        "harmonic_content": 1.0,  # already 0-1
    }

    def _raw_sub_scores(self, features: WindowFeatures) -> dict[str, float]:
        return {
            name: func(features)
            for name, func in self._SUB_SCORE_FUNCS.items()
        }

    def fit(self, all_windows: list[WindowFeatures]) -> None:
        """Compute percentile stats across all windows for normalization."""
        raw_scores: dict[str, list[float]] = {name: [] for name in self._SUB_SCORE_FUNCS}

        for wf in all_windows:
            for name, func in self._SUB_SCORE_FUNCS.items():
                raw_scores[name].append(func(wf))

        for name, values in raw_scores.items():
            arr = np.array(values)
            p5 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            self._percentiles[name] = (p5, p95)

        self._fitted = True

    def calibrate_from_sample(
        self, all_windows: list[WindowFeatures], sample_size: int = 500
    ) -> None:
        """Fit on a random sample for large datasets."""
        if len(all_windows) <= sample_size:
            self.fit(all_windows)
        else:
            sample = random.sample(all_windows, sample_size)
            self.fit(sample)

    def _normalize_sub_score(self, name: str, raw: float) -> float:
        if self.fast:
            divisor = self._FAST_DIVISORS.get(name, 1.0)
            return max(0.0, min(1.0, raw / divisor))

        if not self._fitted:
            raise ValueError(
                "Scorer.fit() must be called before score(). "
                "Use --fast to bypass normalization."
            )

        p5, p95 = self._percentiles[name]
        if p95 - p5 < 1e-10:
            return 0.5
        return max(0.0, min(1.0, (raw - p5) / (p95 - p5)))

    def score(self, features: WindowFeatures) -> ScoredWindow:
        """Score a window and return a ScoredWindow."""
        raw = self._raw_sub_scores(features)
        normalized = {name: self._normalize_sub_score(name, val) for name, val in raw.items()}

        weights = self.config.weights
        weight_map = {
            "spectral_dynamism": weights.spectral_dynamism,
            "textural_richness": weights.textural_richness,
            "timbral_movement": weights.timbral_movement,
            "event_interest": weights.event_interest,
            "dynamic_range": weights.dynamic_range,
            "harmonic_content": weights.harmonic_content,
        }

        final_score = sum(normalized[k] * weight_map[k] for k in normalized)
        final_score = max(0.0, min(1.0, final_score))

        return ScoredWindow(
            source_file=features.source_file,
            start_s=features.start_s,
            end_s=features.end_s,
            features=features,
            score=final_score,
            sub_scores=normalized,
        )
