"""Hard-reject and soft-penalize windows based on quality and content."""

from __future__ import annotations

from collections import defaultdict

from field_miner.models import Config, WindowFeatures


class Filter:
    def __init__(self, config: Config):
        self.config = config
        self._rejection_counts: dict[str, int] = defaultdict(int)

    def hard_reject(self, features: WindowFeatures) -> tuple[bool, str | None]:
        """Check if a window should be hard-rejected. Returns (rejected, reason)."""
        if features.peak_amplitude > self.config.clipping_threshold:
            self._rejection_counts["clipping"] += 1
            return True, "clipping"

        if features.clipping_ratio > 0.005:
            self._rejection_counts["clipping_ratio"] += 1
            return True, "clipping_ratio"

        if features.rms_mean_db < self.config.silence_threshold_dbfs:
            self._rejection_counts["silence"] += 1
            return True, "silence"

        if abs(features.dc_offset) > 0.05:
            self._rejection_counts["dc_offset"] += 1
            return True, "dc_offset"

        # Check top-3 YAMNet labels against reject list
        top3 = features.top_yamnet_labels[:3]
        for label, _score in top3:
            if label in self.config.reject_labels:
                reason = f"rejected_label:{label}"
                self._rejection_counts[reason] += 1
                return True, reason

        return False, None

    def apply_soft_penalties(self, features: WindowFeatures, score: float) -> float:
        """Apply soft score penalties for borderline content."""
        # Labels in top-5 but not top-3 that are in reject list
        if len(features.top_yamnet_labels) > 3:
            for label, _s in features.top_yamnet_labels[3:5]:
                if label in self.config.reject_labels:
                    score *= 0.7
                    break

        # Wind/static/hum penalty
        wind_labels = {"Wind noise", "Static", "Hum"}
        for label, _s in features.top_yamnet_labels[:5]:
            if label in wind_labels:
                score *= self.config.wind_noise_penalty
                break

        # Boring static/drone detection
        if features.onset_count == 0 and features.spectral_centroid_std < 200:
            score *= 0.7

        # Persistent background noise penalty:
        # High spectral flatness = noise-like (flat spectrum / constant din)
        # Low spectral contrast = everything at same level across bands
        # Low RMS variance = constant unchanging level
        if features.spectral_flatness_mean > 0.4 and features.rms_variance < 0.005:
            score *= 0.5  # heavy penalty for flat noisy drone
        elif features.spectral_flatness_mean > 0.25 and features.spectral_centroid_std < 300:
            score *= 0.7  # moderate penalty for steady background noise

        return min(score, 1.0)

    def compute_prefer_boost(self, features: WindowFeatures) -> float:
        """Compute score multiplier for preferred content labels."""
        multiplier = 1.0
        for label, _s in features.top_yamnet_labels[:5]:
            if label in self.config.prefer_labels:
                multiplier *= 1.1
        return min(multiplier, 1.5)

    def rejection_summary(self) -> dict[str, int]:
        return dict(self._rejection_counts)
