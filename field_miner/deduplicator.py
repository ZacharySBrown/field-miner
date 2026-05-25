"""Prune overlapping and near-duplicate windows before export."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rich.console import Console

from field_miner.models import Config, ScoredWindow

console = Console()


class Deduplicator:
    def __init__(self, config: Config):
        self.config = config

    def prune_temporal(self, windows: list[ScoredWindow]) -> list[ScoredWindow]:
        """Remove heavily overlapping windows from the same source file (NMS)."""
        by_file: dict[str, list[ScoredWindow]] = defaultdict(list)
        for w in windows:
            by_file[w.source_file].append(w)

        kept: list[ScoredWindow] = []
        for _file, file_windows in by_file.items():
            file_windows.sort(key=lambda w: w.score, reverse=True)
            selected: list[ScoredWindow] = []
            for candidate in file_windows:
                overlaps = False
                for existing in selected:
                    overlap = self._temporal_overlap(candidate, existing)
                    if overlap > self.config.dedup_temporal_overlap:
                        overlaps = True
                        break
                if not overlaps:
                    selected.append(candidate)
            kept.extend(selected)

        return kept

    @staticmethod
    def _temporal_overlap(a: ScoredWindow, b: ScoredWindow) -> float:
        overlap_start = max(a.start_s, b.start_s)
        overlap_end = min(a.end_s, b.end_s)
        intersection = max(0.0, overlap_end - overlap_start)
        shorter = min(a.end_s - a.start_s, b.end_s - b.start_s)
        if shorter <= 0:
            return 0.0
        return intersection / shorter

    def prune_by_embedding(self, windows: list[ScoredWindow]) -> list[ScoredWindow]:
        """Remove near-duplicate windows across files using YAMNet embeddings."""
        # Split into those with and without embeddings
        with_emb = [w for w in windows if w.features.yamnet_embedding]
        without_emb = [w for w in windows if not w.features.yamnet_embedding]

        if not with_emb:
            return windows

        with_emb.sort(key=lambda w: w.score, reverse=True)

        # For large sets, use approximate comparison
        use_approx = len(with_emb) > 2000
        if use_approx:
            console.print(
                f"[yellow]Large window set ({len(with_emb)}): "
                f"using approximate deduplication[/yellow]"
            )

        kept: list[ScoredWindow] = []
        kept_embeddings: list[np.ndarray] = []

        for candidate in with_emb:
            emb = np.array(candidate.features.yamnet_embedding)
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-10:
                kept.append(candidate)
                kept_embeddings.append(emb)
                continue

            is_dup = False
            # Compare against kept embeddings (last 50 for approx mode)
            compare_embs = kept_embeddings[-50:] if use_approx else kept_embeddings
            for ref_emb in compare_embs:
                ref_norm = np.linalg.norm(ref_emb)
                if ref_norm < 1e-10:
                    continue
                cosine_sim = float(np.dot(emb, ref_emb) / (emb_norm * ref_norm))
                if cosine_sim > self.config.dedup_embedding_similarity:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(candidate)
                kept_embeddings.append(emb)

        return kept + without_emb

    def run(self, windows: list[ScoredWindow]) -> list[ScoredWindow]:
        """Full deduplication pipeline: temporal then embedding."""
        original = len(windows)
        after_temporal = self.prune_temporal(windows)
        after_embedding = self.prune_by_embedding(after_temporal)

        console.print(
            f"[bold]Deduplication:[/bold] {original} → "
            f"{len(after_temporal)} (temporal) → "
            f"{len(after_embedding)} (embedding)"
        )
        return after_embedding
