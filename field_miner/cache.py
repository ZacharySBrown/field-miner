"""Disk cache for analysis results, keyed by file hash + config hash."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from rich.console import Console

from field_miner.models import AudioFile, Config, ScoredWindow

console = Console()


class Cache:
    def __init__(self, config: Config):
        self.cache_dir = Path(config.cache_dir) / "analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config.config_hash()

    def _key_path(self, audio_file: AudioFile) -> Path:
        return self.cache_dir / f"{audio_file.file_hash}_{self.config_hash}.json.gz"

    def has(self, audio_file: AudioFile) -> bool:
        return self._key_path(audio_file).exists()

    def load(self, audio_file: AudioFile) -> list[ScoredWindow]:
        path = self._key_path(audio_file)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.loads(f.read())
        return [ScoredWindow.model_validate(item) for item in data]

    def save(
        self, audio_file: AudioFile, windows: list[ScoredWindow]
    ) -> None:
        path = self._key_path(audio_file)
        data = [w.model_dump(mode="json") for w in windows]
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(data))

    def stats(self) -> dict:
        files = list(self.cache_dir.glob("*.json.gz"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cached_files": len(files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "cache_dir": str(self.cache_dir),
        }
