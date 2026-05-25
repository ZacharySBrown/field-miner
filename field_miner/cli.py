"""CLI entry points for field-miner."""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from field_miner.models import Config, RunManifest, ScoredWindow, WindowFeatures

console = Console()


def _analyze_file_librosa(args: tuple) -> list[dict]:
    """Worker function for parallel librosa feature extraction.

    TensorFlow/YAMNet must NOT run in worker processes — it doesn't fork safely.
    This function only extracts librosa features.
    """
    from field_miner.analyzer import Analyzer
    from field_miner.models import AudioFile, Config

    audio_file_dict, config_dict = args
    config = Config(**config_dict)
    audio_file = AudioFile(**audio_file_dict)
    analyzer = Analyzer(config, use_yamnet=False)
    windows = analyzer.analyze_file(audio_file)
    return [w.model_dump(mode="json") for w in windows]


@click.group()
@click.version_option(package_name="field-miner")
def cli():
    """Field Miner — mine field recordings for interesting audio snippets."""
    console.print(
        Panel.fit(
            "[bold cyan]field-miner[/bold cyan] — "
            "field recording snippet discovery for ambient/IDM production",
            border_style="cyan",
        )
    )


@cli.command()
@click.option("--input", "-i", "input_dir", required=True, type=click.Path(exists=True),
              help="Directory containing field recordings")
@click.option("--output", "-o", "output_dir", default="./mined_clips", type=click.Path(),
              help="Output directory for clips and metadata")
@click.option("--config", "-c", "config_path", default=None, type=click.Path(exists=True),
              help="Path to config.yaml")
@click.option("--fast/--no-fast", default=False, help="Fast mode: skip two-pass normalization")
@click.option("--max-clips", default=None, type=int, help="Override max total clips")
@click.option("--no-yamnet", is_flag=True, help="Disable YAMNet semantic labeling")
@click.option("--dry-run", is_flag=True, help="Report stats without exporting")
@click.option("--workers", default=1, type=int, help="Parallel file processing workers")
@click.option("--resume/--no-resume", default=True, help="Resume from previous partial run")
@click.option("--debug", is_flag=True, help="Show full tracebacks on error")
def analyze(input_dir, output_dir, config_path, fast, max_clips, no_yamnet,
            dry_run, workers, resume, debug):
    """Analyze field recordings and extract interesting snippets."""
    try:
        _run_analyze(input_dir, output_dir, config_path, fast, max_clips,
                     no_yamnet, dry_run, workers, resume)
    except Exception as e:
        if debug:
            traceback.print_exc()
        else:
            console.print(Panel(
                f"[bold red]Error:[/bold red] {e}\n\n"
                "[dim]Run with --debug for full traceback[/dim]",
                border_style="red",
            ))
            sys.exit(1)


def _run_analyze(input_dir, output_dir, config_path, fast, max_clips,
                 no_yamnet, dry_run, workers, resume):
    from field_miner.analyzer import Analyzer
    from field_miner.cache import Cache
    from field_miner.deduplicator import Deduplicator
    from field_miner.exporter import Exporter
    from field_miner.filter import Filter
    from field_miner.ingestor import Ingestor
    from field_miner.scorer import Scorer

    # 1. Load config
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    config.input_dir = input_dir
    config.output_dir = output_dir
    config.fast = fast
    if max_clips is not None:
        config.max_total_clips = max_clips
    if workers:
        config.n_workers = workers

    # Print settings
    console.print(Panel(
        f"[bold]Input:[/bold] {config.input_dir}\n"
        f"[bold]Output:[/bold] {config.output_dir}\n"
        f"[bold]Window:[/bold] {config.window_size}s / hop {config.hop_size}s\n"
        f"[bold]Mode:[/bold] {'fast' if fast else 'standard'} | "
        f"YAMNet: {'off' if no_yamnet else 'on'} | "
        f"Workers: {config.n_workers}",
        title="Run Settings",
    ))

    # 2. Discover files
    ingestor = Ingestor(config)
    audio_files = ingestor.discover()

    if not audio_files:
        console.print("[yellow]No audio files found. Check your input directory.[/yellow]")
        return

    # 3. Check for resume
    cache = Cache(config)
    manifest_path = Path(output_dir) / "run_manifest.json"
    completed_hashes: set[str] = set()

    if resume and manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        completed_hashes = set(manifest.completed_hashes)
        remaining = [f for f in audio_files if f.file_hash not in completed_hashes]
        console.print(
            f"[bold]Resuming:[/bold] {len(completed_hashes)} files complete, "
            f"{len(remaining)} remaining"
        )
    else:
        remaining = audio_files

    # 4. Analyze files
    filt = Filter(config)
    all_window_features: list[WindowFeatures] = []

    # Load cached results for completed files
    for af in audio_files:
        if af.file_hash in completed_hashes and cache.has(af):
            cached = cache.load(af)
            all_window_features.extend([sw.features for sw in cached])

    if remaining:
        if config.n_workers > 1 and len(remaining) > 1:
            # Phase A: Parallel librosa extraction
            console.print(f"[bold]Phase A:[/bold] Parallel librosa extraction ({config.n_workers} workers)")
            args = [
                (af.model_dump(mode="json"), config.model_dump(mode="json"))
                for af in remaining
            ]
            with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
                results = list(pool.map(_analyze_file_librosa, args))

            for af, window_dicts in zip(remaining, results):
                windows = [WindowFeatures.model_validate(d) for d in window_dicts]
                # Apply hard rejection
                valid = []
                for wf in windows:
                    rejected, reason = filt.hard_reject(wf)
                    if rejected:
                        wf.hard_rejected = True
                        wf.reject_reason = reason
                    else:
                        valid.append(wf)
                all_window_features.extend(valid)

            # Phase B: YAMNet in main process (if enabled)
            if not no_yamnet:
                console.print("[bold]Phase B:[/bold] YAMNet semantic labeling (main process)")
                analyzer = Analyzer(config, use_yamnet=True)
                if analyzer.use_yamnet:
                    import librosa as lr
                    for af in remaining:
                        load_path = af.normalized_path or af.path
                        y, sr = lr.load(str(load_path), sr=config.normalize_sr, mono=True)
                        window_samples = int(config.window_size * sr)
                        hop_samples = int(config.hop_size * sr)

                        idx = 0
                        for wf in all_window_features:
                            if wf.source_file != str(af.path):
                                continue
                            if idx % config.yamnet_every_n_windows == 0:
                                start = int(wf.start_s * sr)
                                end = min(start + window_samples, len(y))
                                segment = y[start:end]
                                yamnet_result = analyzer.extract_yamnet_features(segment, sr)
                                wf.top_yamnet_labels = yamnet_result.get("top_yamnet_labels", [])
                                wf.yamnet_embedding = yamnet_result.get("yamnet_embedding", [])
                                last_yamnet = yamnet_result
                            elif idx > 0:
                                wf.top_yamnet_labels = last_yamnet.get("top_yamnet_labels", [])
                                wf.yamnet_embedding = last_yamnet.get("yamnet_embedding", [])
                            idx += 1
        else:
            # Single worker: simple sequential loop
            analyzer = Analyzer(config, use_yamnet=not no_yamnet)
            for af in remaining:
                console.print(f"[bold]Analyzing:[/bold] {af.path.name}")
                windows = analyzer.analyze_file(af)
                valid = []
                for wf in windows:
                    rejected, reason = filt.hard_reject(wf)
                    if rejected:
                        wf.hard_rejected = True
                        wf.reject_reason = reason
                    else:
                        valid.append(wf)
                all_window_features.extend(valid)

    console.print(f"[bold]{len(all_window_features)} valid windows[/bold] from {len(audio_files)} files")

    if not all_window_features:
        console.print("[yellow]No valid windows after filtering.[/yellow]")
        return

    # 5. Score
    scorer = Scorer(config, fast=fast)
    if not fast:
        console.print("[bold]Fitting scorer percentiles...[/bold]")
        scorer.fit(all_window_features)

    scored: list[ScoredWindow] = []
    for wf in all_window_features:
        sw = scorer.score(wf)
        # Apply soft penalties and prefer boost
        sw.score = filt.apply_soft_penalties(wf, sw.score)
        sw.score *= filt.compute_prefer_boost(wf)
        sw.score = min(sw.score, 1.0)
        scored.append(sw)

    # 6. Deduplicate
    deduplicator = Deduplicator(config)
    deduped = deduplicator.run(scored)

    # 7. Save manifest and cache new results
    manifest = RunManifest(
        files=audio_files,
        config_snapshot=config.model_dump(mode="json"),
        completed_hashes=[af.file_hash for af in audio_files],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    for af in remaining:
        file_scored = [sw for sw in deduped if sw.source_file == str(af.path)]
        if file_scored:
            cache.save(af, file_scored)

    # Dry run: stats only
    if dry_run:
        _print_dry_run_stats(deduped, filt)
        return

    # 8. Export
    exporter = Exporter(config)
    exported = exporter.export(deduped)

    # Summary
    console.print(Panel(
        f"[bold green]Done![/bold green]\n"
        f"Clips exported: {sum(1 for e in exported if not e.export_failed)}\n"
        f"Output: {config.output_dir}",
        border_style="green",
    ))


def _print_dry_run_stats(windows: list[ScoredWindow], filt: Filter):
    scores = [w.score for w in windows]
    console.print(Panel(
        f"[bold]Dry Run Results[/bold]\n\n"
        f"Valid windows: {len(windows)}\n"
        f"Score range: {min(scores):.3f} — {max(scores):.3f}\n"
        f"Mean: {np.mean(scores):.3f} | Median: {np.median(scores):.3f}\n"
        f"P90: {np.percentile(scores, 90):.3f}\n\n"
        f"Rejection summary: {filt.rejection_summary()}",
        title="Dry Run",
    ))


@cli.command()
@click.option("--results", required=True, type=click.Path(exists=True),
              help="Path to results.json from a previous run")
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True),
              help="New config.yaml with updated weights")
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(),
              help="New output directory")
def rescore(results, config_path, output_dir):
    """Re-score and re-export using different weights (no re-analysis)."""
    from field_miner.deduplicator import Deduplicator
    from field_miner.exporter import Exporter
    from field_miner.filter import Filter
    from field_miner.scorer import Scorer

    config = Config.from_yaml(config_path)
    config.output_dir = output_dir

    with open(results) as f:
        data = json.load(f)

    # Reconstruct ScoredWindows from exported clips — we need features
    console.print(f"[bold]Loading {len(data)} clips from previous run...[/bold]")

    # For rescore, we need the cached analysis data, not just export metadata.
    # Check if there's a cache we can load from
    console.print("[yellow]Note: rescore works best with cached analysis data. "
                  "Re-scoring from export metadata uses limited features.[/yellow]")

    scorer = Scorer(config, fast=True)
    filt = Filter(config)
    deduplicator = Deduplicator(config)
    exporter = Exporter(config)

    # Reconstruct minimal WindowFeatures from export data
    windows: list[WindowFeatures] = []
    for item in data:
        wf = WindowFeatures(
            source_file=item["source_file"],
            start_s=item["start_s"],
            end_s=item["end_s"],
            spectral_centroid_mean=item.get("spectral_centroid_mean", 0),
            rms_mean_db=item.get("rms_mean_db", 0),
            onset_count=item.get("onset_count", 0),
            mfcc_delta_std=item.get("mfcc_delta_std", 0),
            harmonic_ratio=item.get("harmonic_ratio", 0),
            top_yamnet_labels=[(l, s) for l, s in item.get("top_labels", [])],
        )
        windows.append(wf)

    scored = [scorer.score(wf) for wf in windows]
    for sw in scored:
        sw.score = filt.apply_soft_penalties(sw.features, sw.score)
        sw.score *= filt.compute_prefer_boost(sw.features)
        sw.score = min(sw.score, 1.0)

    deduped = deduplicator.run(scored)
    exported = exporter.export(deduped)

    console.print(f"[bold green]Rescore complete: {len(exported)} clips → {output_dir}[/bold green]")


@cli.command()
@click.option("--results", required=True, type=click.Path(exists=True),
              help="Path to results.json")
def stats(results):
    """Show statistics on a completed run."""
    with open(results) as f:
        data = json.load(f)

    if not data:
        console.print("[yellow]No results found.[/yellow]")
        return

    scores = [item["score"] for item in data]
    table = Table(title="Run Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total clips", str(len(data)))
    table.add_row("Score min", f"{min(scores):.4f}")
    table.add_row("Score max", f"{max(scores):.4f}")
    table.add_row("Score mean", f"{np.mean(scores):.4f}")
    table.add_row("Score median", f"{np.median(scores):.4f}")
    table.add_row("Score P90", f"{np.percentile(scores, 90):.4f}")
    console.print(table)

    # Top labels
    from collections import Counter
    label_counts: Counter = Counter()
    for item in data:
        for label, _score in item.get("top_labels", []):
            label_counts[label] += 1

    if label_counts:
        label_table = Table(title="Top Labels")
        label_table.add_column("Label", style="cyan")
        label_table.add_column("Count", justify="right")
        for label, count in label_counts.most_common(10):
            label_table.add_row(label, str(count))
        console.print(label_table)

    # Per-file breakdown
    from collections import defaultdict
    per_file: dict[str, list] = defaultdict(list)
    for item in data:
        per_file[Path(item["source_file"]).name].append(item["score"])

    file_table = Table(title="Per-File Breakdown")
    file_table.add_column("File", style="cyan")
    file_table.add_column("Clips", justify="right")
    file_table.add_column("Avg Score", justify="right")
    for name, file_scores in sorted(per_file.items()):
        file_table.add_row(name, str(len(file_scores)), f"{np.mean(file_scores):.3f}")
    console.print(file_table)


@cli.command()
@click.option("--results", required=True, type=click.Path(exists=True),
              help="Path to results.json")
@click.option("--top", default=20, type=int, help="Number of top clips to show")
@click.option("--play", is_flag=True, help="Play clips (requires mpv or afplay)")
def review(results, top, play):
    """Review top clips from a completed run."""
    import shutil
    import subprocess

    with open(results) as f:
        data = json.load(f)

    data.sort(key=lambda x: x["score"], reverse=True)
    data = data[:top]

    table = Table(title=f"Top {top} Clips")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("File", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Labels")

    for i, item in enumerate(data, 1):
        labels = "; ".join(l for l, _ in item.get("top_labels", [])[:3])
        table.add_row(
            str(i),
            f"{item['score']:.3f}",
            Path(item.get("clip_path", item["source_file"])).name,
            f"{item['start_s']:.1f}s",
            f"{item['duration_s']:.1f}s",
            labels or "—",
        )

    console.print(table)

    if play:
        player = shutil.which("mpv") or shutil.which("afplay")
        if not player:
            console.print("[yellow]No audio player found (install mpv or use macOS afplay)[/yellow]")
            return

        for i, item in enumerate(data, 1):
            clip_path = item.get("clip_path", "")
            if not clip_path or not Path(clip_path).exists():
                continue
            console.print(f"\n[bold]Playing {i}/{len(data)}:[/bold] {Path(clip_path).name} "
                          f"(score: {item['score']:.3f})")
            try:
                subprocess.run([player, clip_path], timeout=120)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped playback[/dim]")
                break


if __name__ == "__main__":
    cli()
