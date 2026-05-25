"""Slice and export final clips with metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from rich.console import Console
from tqdm import tqdm

from field_miner.models import Config, ExportedClip, ScoredWindow

console = Console()


class Exporter:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.clips_dir = self.output_dir / "clips"

    def export(self, windows: list[ScoredWindow]) -> list[ExportedClip]:
        """Select, slice, and export top clips with metadata."""
        # Step 1: Selection
        candidates = [w for w in windows if w.score >= self.config.min_score_threshold]
        candidates.sort(key=lambda w: w.score, reverse=True)

        # Per-file cap
        per_file_count: dict[str, int] = defaultdict(int)
        selected: list[ScoredWindow] = []
        for w in candidates:
            if per_file_count[w.source_file] >= self.config.max_clips_per_file:
                continue
            if len(selected) >= self.config.max_total_clips:
                break
            selected.append(w)
            per_file_count[w.source_file] += 1

        console.print(
            f"[bold]Export selection:[/bold] {len(candidates)} above threshold, "
            f"{len(selected)} after caps"
        )

        if not selected:
            console.print("[yellow]No clips to export.[/yellow]")
            return []

        # Create output directories
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Slice clips
        exported: list[ExportedClip] = []
        for sw in tqdm(selected, desc="Exporting clips"):
            try:
                clip_path = self._slice_clip(sw)
                clip = ExportedClip.from_scored_window(sw, str(clip_path))
                exported.append(clip)

                if self.config.ableton_export:
                    self._write_ableton_sidecar(clip, clip_path)
            except Exception as e:
                console.print(f"[yellow]Export failed for {sw.source_file} @ {sw.start_s:.1f}s: {e}[/yellow]")
                clip = ExportedClip.from_scored_window(sw, "")
                clip.export_failed = True
                exported.append(clip)

        # Step 3: Metadata
        self._write_metadata(exported, windows)

        # Step 4: Score histogram
        self._write_histogram(windows)

        # Step 5: Run log
        self._write_run_log(windows, exported, selected)

        console.print(
            f"[bold green]Exported {sum(1 for e in exported if not e.export_failed)} clips "
            f"to {self.clips_dir}[/bold green]"
        )
        return exported

    def _slice_clip(self, sw: ScoredWindow) -> Path:
        """Slice a clip from the source audio file."""
        source_path = Path(sw.source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file missing: {source_path}")

        data, sr = sf.read(str(source_path), dtype="float64")

        # Handle stereo → use as-is for export
        if data.ndim > 1:
            pass  # keep multichannel

        start_sample = int(sw.start_s * sr)
        end_sample = int(sw.end_s * sr)
        segment = data[start_sample:end_sample]

        # Apply fades
        fade_in_samples = int(self.config.fade_in_ms * sr / 1000)
        fade_out_samples = int(self.config.fade_out_ms * sr / 1000)

        if segment.ndim == 1:
            if fade_in_samples > 0 and fade_in_samples < len(segment):
                segment[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
            if fade_out_samples > 0 and fade_out_samples < len(segment):
                segment[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
        else:
            if fade_in_samples > 0 and fade_in_samples < len(segment):
                fade_in = np.linspace(0, 1, fade_in_samples)[:, np.newaxis]
                segment[:fade_in_samples] *= fade_in
            if fade_out_samples > 0 and fade_out_samples < len(segment):
                fade_out = np.linspace(1, 0, fade_out_samples)[:, np.newaxis]
                segment[-fade_out_samples:] *= fade_out

        # Optional normalization to -6 dBFS peak
        if self.config.normalize_output:
            peak = np.max(np.abs(segment))
            if peak > 0:
                target = 10 ** (-6 / 20)  # -6 dBFS
                segment = segment * (target / peak)

        # Build filename
        stem = source_path.stem
        start_ms = int(sw.start_s * 1000)
        score_pct = int(sw.score * 100)
        filename = f"{stem}_{start_ms:07d}ms_{score_pct:03d}.wav"
        out_path = self.clips_dir / filename

        sf.write(str(out_path), segment, sr, subtype="PCM_24")
        return out_path

    def _write_metadata(self, exported: list[ExportedClip], all_windows: list[ScoredWindow]) -> None:
        """Write results.csv and results.json."""
        # JSON
        json_path = self.output_dir / "results.json"
        data = [clip.model_dump(mode="json") for clip in exported if not clip.export_failed]
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # CSV
        csv_path = self.output_dir / "results.csv"
        rows = []
        for clip in exported:
            if clip.export_failed:
                continue
            labels_str = "; ".join(f"{l}({s:.2f})" for l, s in clip.top_labels)
            rows.append({
                "source_file": clip.source_file,
                "start_time_s": clip.start_s,
                "end_time_s": clip.end_s,
                "duration_s": clip.duration_s,
                "score": round(clip.score, 4),
                "top_labels": labels_str,
                "spectral_centroid_mean": round(clip.spectral_centroid_mean, 2),
                "rms_mean_db": round(clip.rms_mean_db, 2),
                "onset_count": clip.onset_count,
                "mfcc_delta_std": round(clip.mfcc_delta_std, 4),
                "harmonic_ratio": round(clip.harmonic_ratio, 4),
                "clip_path": clip.clip_path,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def _write_histogram(self, all_windows: list[ScoredWindow]) -> None:
        """Write score distribution histogram if matplotlib available."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            scores = [w.score for w in all_windows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(scores, bins=50, color="#4a90d9", edgecolor="white", alpha=0.8)
            ax.axvline(
                x=self.config.min_score_threshold,
                color="red", linestyle="--", linewidth=2,
                label=f"Threshold ({self.config.min_score_threshold})",
            )
            ax.set_xlabel("Interestingness Score")
            ax.set_ylabel("Window Count")
            ax.set_title("Score Distribution")
            ax.legend()
            fig.savefig(self.output_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            console.print("[dim]matplotlib not available, skipping histogram[/dim]")

    def _write_run_log(
        self,
        all_windows: list[ScoredWindow],
        exported: list[ExportedClip],
        selected: list[ScoredWindow],
    ) -> None:
        """Write a human-readable run log."""
        log_path = self.output_dir / "run_log.txt"
        successful = sum(1 for e in exported if not e.export_failed)
        failed = sum(1 for e in exported if e.export_failed)

        source_files = set(w.source_file for w in all_windows)
        scores = [w.score for w in all_windows]

        lines = [
            f"Field Miner Run Log — {datetime.now().isoformat()}",
            f"",
            f"Source files analyzed: {len(source_files)}",
            f"Total windows analyzed: {len(all_windows)}",
            f"Windows above threshold: {len(selected)}",
            f"Clips exported: {successful}",
            f"Export failures: {failed}",
            f"",
            f"Score stats:",
            f"  Min:    {min(scores):.4f}" if scores else "  No scores",
            f"  Max:    {max(scores):.4f}" if scores else "",
            f"  Mean:   {np.mean(scores):.4f}" if scores else "",
            f"  Median: {np.median(scores):.4f}" if scores else "",
            f"  P90:    {np.percentile(scores, 90):.4f}" if scores else "",
        ]

        with open(log_path, "w") as f:
            f.write("\n".join(lines))

    def _write_ableton_sidecar(self, clip: ExportedClip, clip_path: Path) -> None:
        """Write minimal Ableton .asd sidecar for clip color tagging."""
        color_map = {
            "Water": 13, "Stream": 13, "Rain": 13, "Waves, surf": 13, "Waterfall": 13,
            "Bird": 9, "Birdsong": 9,
            "Wind": 0, "Rustling leaves": 0,
            "Insect": 14, "Cricket": 14, "Frog": 14,
            "Fire": 4, "Thunder": 4,
        }

        color_id = 1  # default white
        for label, _score in clip.top_labels:
            if label in color_map:
                color_id = color_map[label]
                break

        # Read back to get sample count and sr
        try:
            info = sf.info(str(clip_path))
            duration_samples = info.frames
            sample_rate = info.samplerate
        except Exception:
            return

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Ableton MajorVersion="5" MinorVersion="10.0.2" '
            'SchemaChangeCount="3" Creator="field-miner" Revision="">\n'
            "  <SampleData>\n"
            "    <SampleRef>\n"
            f"      <FileRef><Name>{clip_path.name}</Name></FileRef>\n"
            "    </SampleRef>\n"
            f"    <DefaultDuration>{duration_samples}</DefaultDuration>\n"
            f"    <DefaultSampleRate>{sample_rate}</DefaultSampleRate>\n"
            f"    <Color>{color_id}</Color>\n"
            "  </SampleData>\n"
            "</Ableton>\n"
        )
        asd_path = clip_path.with_suffix(clip_path.suffix + ".asd")
        with open(asd_path, "w") as f:
            f.write(xml)
