"""Discover, validate, and normalize audio files for analysis."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import soundfile as sf
from rich.console import Console
from rich.table import Table

from field_miner.models import AudioFile, Config

console = Console()

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


class Ingestor:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.cache_dir = Path(config.cache_dir) / "normalized"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def discover(self) -> list[AudioFile]:
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found on PATH. Install it:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg"
            )

        files: list[AudioFile] = []
        pattern = self.config.include_pattern

        for path in sorted(self.input_dir.rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if pattern and not path.match(pattern):
                continue

            try:
                audio_file = self._process_file(path)
                if audio_file:
                    files.append(audio_file)
            except Exception as e:
                console.print(f"[yellow]Warning: skipping {path.name}: {e}[/yellow]")

        self._print_summary(files)
        return files

    def _process_file(self, path: Path) -> AudioFile | None:
        try:
            info = sf.info(str(path))
        except Exception:
            # Try ffprobe for formats soundfile can't read directly
            info = self._probe_with_ffmpeg(path)
            if info is None:
                console.print(f"[yellow]Warning: cannot read {path.name}, skipping[/yellow]")
                return None

        duration = info.duration if hasattr(info, "duration") else info["duration"]
        sr = info.samplerate if hasattr(info, "samplerate") else info["samplerate"]
        channels = info.channels if hasattr(info, "channels") else info["channels"]

        if duration < self.config.min_file_duration:
            console.print(
                f"[dim]Skipping {path.name}: {duration:.1f}s < {self.config.min_file_duration}s minimum[/dim]"
            )
            return None

        file_hash = self._hash_file(path)
        normalized_path = self._normalize(path, file_hash) if path.suffix.lower() != ".wav" else None

        return AudioFile(
            path=path,
            normalized_path=normalized_path,
            duration_s=duration,
            sample_rate=sr,
            channels=channels,
            size_bytes=path.stat().st_size,
            file_hash=file_hash,
        )

    def _probe_with_ffmpeg(self, path: Path) -> dict | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-show_entries", "stream=sample_rate,channels",
                    "-of", "csv=p=0", str(path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                stream_parts = lines[0].split(",")
                duration = float(lines[1].strip())
                return {
                    "duration": duration,
                    "samplerate": int(stream_parts[0]),
                    "channels": int(stream_parts[1]),
                }
        except Exception:
            pass
        return None

    def _hash_file(self, path: Path) -> str:
        """Hash first 4MB of file for speed."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
        return h.hexdigest()

    def _normalize(self, path: Path, file_hash: str) -> Path:
        """Convert non-WAV to normalized WAV via ffmpeg."""
        out_path = self.cache_dir / f"{file_hash}.wav"
        if out_path.exists():
            return out_path

        cmd = [
            "ffmpeg", "-i", str(path),
            "-ar", str(self.config.normalize_sr),
            "-ac", str(self.config.normalize_channels),
            "-sample_fmt", "s16",
            str(out_path),
            "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        return out_path

    def _print_summary(self, files: list[AudioFile]) -> None:
        table = Table(title="Discovered Audio Files")
        table.add_column("File", style="cyan")
        table.add_column("Duration", justify="right")
        table.add_column("SR", justify="right")
        table.add_column("Ch", justify="right")
        table.add_column("Size", justify="right")

        total_duration = 0.0
        for af in files:
            total_duration += af.duration_s
            table.add_row(
                af.path.name,
                f"{af.duration_s:.1f}s",
                str(af.sample_rate),
                str(af.channels),
                f"{af.size_bytes / 1024 / 1024:.1f}MB",
            )

        console.print(table)
        console.print(
            f"[bold green]{len(files)} files[/bold green], "
            f"[bold]{total_duration / 60:.1f} minutes[/bold] total"
        )
