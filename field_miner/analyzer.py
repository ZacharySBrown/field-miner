"""Slide a window across audio files and extract feature vectors."""

from __future__ import annotations

import platform
import warnings
from typing import Any

import librosa
import numpy as np
from rich.console import Console
from scipy import stats
from tqdm import tqdm

from field_miner.models import AudioFile, Config, WindowFeatures

console = Console()


class Analyzer:
    def __init__(self, config: Config, use_yamnet: bool = True):
        self.config = config
        self.use_yamnet = use_yamnet
        self.yamnet_model = None
        self.yamnet_class_names: list[str] = []

        if self.use_yamnet:
            self._load_yamnet()

    def _load_yamnet(self) -> None:
        try:
            import tensorflow_hub as hub

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                console.print(
                    "[dim]Apple Silicon detected. If TF is slow, try: "
                    "pip install tensorflow-metal[/dim]"
                )

            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            # Load class names from the model
            import tensorflow as tf
            import csv

            class_map_path = self.yamnet_model.class_map_path().numpy().decode("utf-8")
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self.yamnet_class_names = [row["display_name"] for row in reader]

            console.print("[green]YAMNet loaded successfully[/green]")
        except Exception as e:
            console.print(
                f"[yellow]YAMNet unavailable ({e}). "
                f"Running without semantic labels.[/yellow]"
            )
            self.use_yamnet = False

    def extract_librosa_features(self, segment: np.ndarray, sr: int) -> dict[str, Any]:
        """Extract all librosa-based acoustic and signal quality features."""
        features: dict[str, Any] = {}

        # Signal quality
        features["peak_amplitude"] = float(np.max(np.abs(segment))) if len(segment) > 0 else 0.0
        features["dc_offset"] = float(np.mean(segment)) if len(segment) > 0 else 0.0

        rms = librosa.feature.rms(y=segment)[0]
        rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
        features["rms_mean_db"] = float(20 * np.log10(rms_mean + 1e-10))

        above_threshold = np.sum(np.abs(segment) > 0.98)
        features["clipping_ratio"] = float(above_threshold / max(len(segment), 1))

        # Spectral centroid
        try:
            sc = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(sc))
            features["spectral_centroid_std"] = float(np.std(sc))
        except Exception:
            features["spectral_centroid_mean"] = 0.0
            features["spectral_centroid_std"] = 0.0

        # Spectral rolloff
        try:
            sr_feat = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
            features["spectral_rolloff_mean"] = float(np.mean(sr_feat))
            features["spectral_rolloff_std"] = float(np.std(sr_feat))
        except Exception:
            features["spectral_rolloff_mean"] = 0.0
            features["spectral_rolloff_std"] = 0.0

        # Spectral contrast
        try:
            sc_bands = librosa.feature.spectral_contrast(y=segment, sr=sr)
            features["spectral_contrast"] = [float(np.mean(band)) for band in sc_bands]
        except Exception:
            features["spectral_contrast"] = [0.0] * 7

        # Spectral flatness
        try:
            sf_feat = librosa.feature.spectral_flatness(y=segment)[0]
            features["spectral_flatness_mean"] = float(np.mean(sf_feat))
            features["spectral_flatness_std"] = float(np.std(sf_feat))
        except Exception:
            features["spectral_flatness_mean"] = 0.0
            features["spectral_flatness_std"] = 0.0

        # MFCC
        try:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = [float(np.mean(c)) for c in mfcc]
            features["mfcc_std"] = [float(np.std(c)) for c in mfcc]
            # MFCC delta
            mfcc_delta = librosa.feature.delta(mfcc)
            features["mfcc_delta_std"] = float(np.mean([np.std(d) for d in mfcc_delta]))
        except Exception:
            features["mfcc_mean"] = [0.0] * 13
            features["mfcc_std"] = [0.0] * 13
            features["mfcc_delta_std"] = 0.0

        # Zero-crossing rate
        try:
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))
        except Exception:
            features["zcr_mean"] = 0.0
            features["zcr_std"] = 0.0

        # Onset detection
        try:
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            features["onset_strength_mean"] = float(np.mean(onset_env))
            features["onset_strength_std"] = float(np.std(onset_env))
            onsets = librosa.onset.onset_detect(y=segment, sr=sr)
            features["onset_count"] = int(len(onsets))
        except Exception:
            features["onset_strength_mean"] = 0.0
            features["onset_strength_std"] = 0.0
            features["onset_count"] = 0

        # RMS variance (over 8 sub-windows)
        try:
            n_sub = 8
            sub_len = len(segment) // n_sub
            if sub_len > 0:
                sub_rms = [
                    float(np.sqrt(np.mean(segment[i * sub_len : (i + 1) * sub_len] ** 2)))
                    for i in range(n_sub)
                ]
                features["rms_variance"] = float(np.std(sub_rms))
            else:
                features["rms_variance"] = 0.0
        except Exception:
            features["rms_variance"] = 0.0

        # Mel spectrogram entropy
        try:
            mel = librosa.feature.melspectrogram(y=segment, sr=sr)
            mel_mean = np.mean(mel, axis=1)
            mel_mean = mel_mean / (np.sum(mel_mean) + 1e-10)
            features["mel_entropy"] = float(stats.entropy(mel_mean + 1e-10))
        except Exception:
            features["mel_entropy"] = 0.0

        # Harmonic ratio
        try:
            harmonic, percussive = librosa.effects.hpss(segment)
            h_energy = float(np.sum(harmonic**2))
            total_energy = h_energy + float(np.sum(percussive**2))
            features["harmonic_ratio"] = h_energy / (total_energy + 1e-10)
        except Exception:
            features["harmonic_ratio"] = 0.0

        return features

    def extract_yamnet_features(
        self, segment: np.ndarray, sr: int
    ) -> dict[str, Any]:
        """Run YAMNet on a segment, return top labels and embedding."""
        if not self.use_yamnet or self.yamnet_model is None:
            return {"top_yamnet_labels": [], "yamnet_embedding": []}

        try:
            import tensorflow as tf

            # Resample to 16kHz mono
            if sr != 16000:
                segment_16k = librosa.resample(segment, orig_sr=sr, target_sr=16000)
            else:
                segment_16k = segment

            waveform = tf.cast(segment_16k, tf.float32)
            scores, embeddings, spectrogram = self.yamnet_model(waveform)

            # Mean-pool across frames
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()

            # Top 5 labels
            top_indices = np.argsort(mean_scores)[-5:][::-1]
            top_labels = [
                (self.yamnet_class_names[i], float(mean_scores[i]))
                for i in top_indices
                if i < len(self.yamnet_class_names)
            ]

            return {
                "top_yamnet_labels": top_labels,
                "yamnet_embedding": [float(x) for x in mean_embedding],
            }
        except Exception as e:
            warnings.warn(f"YAMNet inference failed: {e}")
            return {"top_yamnet_labels": [], "yamnet_embedding": []}

    def analyze_file(self, audio_file: AudioFile) -> list[WindowFeatures]:
        """Analyze a full audio file, returning features for each window."""
        load_path = audio_file.normalized_path or audio_file.path

        y, sr = librosa.load(str(load_path), sr=self.config.normalize_sr, mono=True)
        total_samples = len(y)
        window_samples = int(self.config.window_size * sr)
        hop_samples = int(self.config.hop_size * sr)

        windows: list[WindowFeatures] = []
        last_yamnet: dict[str, Any] = {"top_yamnet_labels": [], "yamnet_embedding": []}
        window_idx = 0

        positions = list(range(0, total_samples - window_samples + 1, hop_samples))

        for start_sample in tqdm(
            positions,
            desc=f"  {audio_file.path.name}",
            leave=False,
        ):
            end_sample = start_sample + window_samples
            segment = y[start_sample:end_sample]

            if len(segment) < sr:  # skip windows shorter than 1s
                continue

            start_s = start_sample / sr
            end_s = end_sample / sr

            # Librosa features
            try:
                features = self.extract_librosa_features(segment, sr)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: feature extraction failed at "
                    f"{start_s:.1f}s in {audio_file.path.name}: {e}[/yellow]"
                )
                features = {}

            # YAMNet: run every Nth window
            if self.use_yamnet and window_idx % self.config.yamnet_every_n_windows == 0:
                yamnet_result = self.extract_yamnet_features(segment, sr)
                if yamnet_result["top_yamnet_labels"]:
                    last_yamnet = yamnet_result

            features.update(last_yamnet)

            wf = WindowFeatures(
                source_file=str(audio_file.path),
                start_s=start_s,
                end_s=end_s,
                **{k: v for k, v in features.items() if k in WindowFeatures.model_fields},
            )
            windows.append(wf)
            window_idx += 1

        return windows
