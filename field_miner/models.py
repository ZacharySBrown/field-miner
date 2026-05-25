"""Core data models for field-miner."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AudioFile(BaseModel):
    """Represents a discovered input audio file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    normalized_path: Path | None = None
    duration_s: float
    sample_rate: int
    channels: int
    size_bytes: int
    file_hash: str


class WindowFeatures(BaseModel):
    """All features extracted from one analysis window."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: str
    start_s: float
    end_s: float

    # Signal quality
    peak_amplitude: float = 0.0
    rms_mean_db: float = 0.0
    clipping_ratio: float = 0.0
    dc_offset: float = 0.0

    # Spectral
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_rolloff_std: float = 0.0
    spectral_contrast: list[float] = Field(default_factory=list)
    spectral_flatness_mean: float = 0.0
    spectral_flatness_std: float = 0.0

    # MFCC
    mfcc_mean: list[float] = Field(default_factory=list)
    mfcc_std: list[float] = Field(default_factory=list)
    mfcc_delta_std: float = 0.0

    # Rhythm / dynamics
    zcr_mean: float = 0.0
    zcr_std: float = 0.0
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    onset_count: int = 0
    rms_variance: float = 0.0

    # Spectral complexity
    mel_entropy: float = 0.0
    harmonic_ratio: float = 0.0

    # YAMNet semantic
    top_yamnet_labels: list[tuple[str, float]] = Field(default_factory=list)
    yamnet_embedding: list[float] = Field(default_factory=list)

    # Status
    hard_rejected: bool = False
    reject_reason: str | None = None
    analysis_failed: bool = False


class ScoredWindow(BaseModel):
    """A window with its composite interestingness score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: str
    start_s: float
    end_s: float

    # All features carried forward
    features: WindowFeatures

    # Scoring
    score: float = 0.0
    sub_scores: dict[str, float] = Field(default_factory=dict)


class ExportedClip(BaseModel):
    """A scored window that has been exported as an audio clip."""

    source_file: str
    start_s: float
    end_s: float
    duration_s: float
    score: float
    sub_scores: dict[str, float] = Field(default_factory=dict)
    top_labels: list[tuple[str, float]] = Field(default_factory=list)
    spectral_centroid_mean: float = 0.0
    rms_mean_db: float = 0.0
    onset_count: int = 0
    mfcc_delta_std: float = 0.0
    harmonic_ratio: float = 0.0
    clip_path: str = ""
    export_timestamp: datetime = Field(default_factory=datetime.now)
    export_failed: bool = False

    @classmethod
    def from_scored_window(cls, sw: ScoredWindow, clip_path: str) -> ExportedClip:
        return cls(
            source_file=sw.source_file,
            start_s=sw.start_s,
            end_s=sw.end_s,
            duration_s=sw.end_s - sw.start_s,
            score=sw.score,
            sub_scores=sw.sub_scores,
            top_labels=sw.features.top_yamnet_labels,
            spectral_centroid_mean=sw.features.spectral_centroid_mean,
            rms_mean_db=sw.features.rms_mean_db,
            onset_count=sw.features.onset_count,
            mfcc_delta_std=sw.features.mfcc_delta_std,
            harmonic_ratio=sw.features.harmonic_ratio,
            clip_path=clip_path,
        )


class RunManifest(BaseModel):
    """Record of all files processed in a run."""

    files: list[AudioFile] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    completed_hashes: list[str] = Field(default_factory=list)


class WeightsConfig(BaseModel):
    """Scoring weight configuration."""

    spectral_dynamism: float = 0.25
    textural_richness: float = 0.20
    timbral_movement: float = 0.20
    event_interest: float = 0.15
    dynamic_range: float = 0.10
    harmonic_content: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> WeightsConfig:
        total = (
            self.spectral_dynamism
            + self.textural_richness
            + self.timbral_movement
            + self.event_interest
            + self.dynamic_range
            + self.harmonic_content
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return self


class Config(BaseModel):
    """Full application configuration."""

    # Paths
    input_dir: str = "./recordings"
    output_dir: str = "./mined_clips"
    cache_dir: str = "./.cache"

    # Windowing
    window_size: int = 20
    hop_size: int = 5
    min_clip_duration: int = 10
    max_clip_duration: int = 60

    # File filtering
    min_file_duration: int = 30
    normalize_sr: int = 44100
    normalize_channels: int = 1
    include_pattern: str | None = None

    # Scoring
    weights: WeightsConfig = Field(default_factory=WeightsConfig)

    # Filtering
    silence_threshold_dbfs: float = -60.0
    clipping_threshold: float = 0.99
    reject_labels: list[str] = Field(
        default_factory=lambda: [
            "Speech", "Conversation", "Narration, monologue", "Singing",
            "Screaming", "Child speech", "Traffic noise", "Car",
            "Motorcycle", "Siren", "Air horn", "Truck",
            "Static", "White noise", "Hum",
        ]
    )
    prefer_labels: list[str] = Field(
        default_factory=lambda: [
            "Bird", "Birdsong", "Water", "Stream", "Rain", "Wind",
            "Rustling leaves", "Insect", "Cricket", "Frog",
            "Waves, surf", "Fire", "Thunder", "Waterfall",
            "Forest", "Ambient music",
        ]
    )
    wind_noise_penalty: float = 0.6

    # Deduplication
    dedup_temporal_overlap: float = 0.5
    dedup_embedding_similarity: float = 0.92

    # Export
    max_clips_per_file: int = 10
    max_total_clips: int = 500
    min_score_threshold: float = 0.50
    normalize_output: bool = False
    fade_in_ms: int = 50
    fade_out_ms: int = 100
    score_loops: bool = False
    ableton_export: bool = False

    # Performance
    yamnet_every_n_windows: int = 3
    n_workers: int = 4
    fast: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def config_hash(self) -> str:
        """Hash of windowing params for cache keying."""
        key = f"{self.window_size}:{self.hop_size}:{self.normalize_sr}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
