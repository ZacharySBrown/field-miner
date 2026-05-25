# Field Recording Miner — Claude Code Implementation Guide

A step-by-step guide for building the `field-miner` CLI application using Claude Code as your agentic development harness. This guide is structured as a series of focused sessions, each with exact prompts, expected outputs, and validation steps.

---

## Prerequisites

Before starting, make sure you have:

- Claude Code installed and authenticated (`npm install -g @anthropic-ai/claude-code`)
- Python 3.11+ with `pip`
- `ffmpeg` installed and on PATH (`brew install ffmpeg` on Mac)
- A virtual environment ready (or use `uv` — recommended)
- The `field-recording-miner-spec.md` file in your project directory

```bash
mkdir field-miner && cd field-miner
python -m venv .venv && source .venv/bin/activate
# OR with uv (faster):
uv venv && source .venv/bin/activate
```

---

## How to Use This Guide

Each section is a **Claude Code session** with:
- A **context block** to paste at the start of the session
- **Prompts** to send in sequence
- **Validation steps** to run yourself before moving on
- **Gotchas** — known issues to watch for with this stack

The key principle: **one session per module**. Keep Claude Code sessions focused on a single file or concern. Long multi-file sessions accumulate drift. When a session completes a module, validate it, commit it, then start fresh.

---

## Session 0 — Project Scaffold

**Goal**: Create the full project structure, `pyproject.toml`, and `config.yaml` before writing any logic.

### Prompt 0.1 — Scaffold

```
Read the attached spec file: field-recording-miner-spec.md

Then scaffold a Python CLI project called `field-miner` with this exact structure:

field-miner/
  field_miner/
    __init__.py
    cli.py           # click CLI entry points
    ingestor.py      # placeholder
    analyzer.py      # placeholder
    scorer.py        # placeholder
    filter.py        # placeholder
    deduplicator.py  # placeholder
    exporter.py      # placeholder
    cache.py         # placeholder
    models.py        # shared Pydantic dataclasses
  tests/
    __init__.py
    fixtures/        # empty for now
    test_filter.py   # placeholder
    test_scorer.py   # placeholder
  config.yaml        # full default config from spec
  pyproject.toml     # with all dependencies from spec
  README.md          # one-paragraph summary
  .gitignore

For pyproject.toml, use these exact dependencies:
  librosa>=0.10
  soundfile>=0.12
  pydub>=0.25
  tensorflow>=2.15
  tensorflow-hub>=0.16
  numpy>=1.26
  scipy>=1.12
  tqdm>=4.66
  rich>=13.7
  click>=8.1
  pandas>=2.2
  pedalboard>=0.9
  pydantic>=2.6

Entry point: `field-miner = field_miner.cli:cli`

For each placeholder module, add a module docstring explaining its role and a single `pass` or stub class. Do not implement any logic yet.
```

### Prompt 0.2 — Models

```
Now implement field_miner/models.py with Pydantic v2 dataclasses for the core data structures.

Required models:
- `AudioFile` — represents a discovered input file (path, duration_s, sample_rate, channels, size_bytes, file_hash)
- `WindowFeatures` — all features extracted from one window (source_file, start_s, end_s, all acoustic features as typed floats/lists, top_yamnet_labels as list[tuple[str, float]], hard_rejected: bool, reject_reason: str | None)
- `ScoredWindow` — WindowFeatures + score: float + sub_scores: dict[str, float]
- `ExportedClip` — ScoredWindow + clip_path: str + export_timestamp: datetime
- `RunManifest` — list of AudioFile + run metadata (start_time, config snapshot)
- `Config` — full config schema matching config.yaml, with field validators

Use `model_config = ConfigDict(arbitrary_types_allowed=True)` where needed for numpy array fields. Numpy arrays should be typed as `Any` with a docstring comment.

All models should have `model_json_schema()` compatible types — no raw numpy in exported fields.
```

### Validation

```bash
pip install -e ".[dev]"
python -c "from field_miner.models import Config, ScoredWindow; print('Models OK')"
field-miner --help
```

---

## Session 1 — Ingestor

**Goal**: Discover and normalize audio files, produce a `RunManifest`.

### Context block (paste at session start)

```
Project: field-miner — a CLI tool that mines field recordings for interesting ambient music snippets.
Spec: see field-recording-miner-spec.md
Current state: scaffold and models are complete.
Today's task: implement field_miner/ingestor.py and field_miner/cache.py
```

### Prompt 1.1 — Ingestor core

```
Implement field_miner/ingestor.py

Requirements from spec:
- Recursively discover .wav, .aif, .aiff, .flac, .mp3 files under input_dir
- Convert non-WAV formats to temporary normalized WAV using ffmpeg subprocess (16-bit, configurable sample rate and channels)
- Skip files shorter than config.min_file_duration
- Compute MD5 hash of each file (for cache keying) — only hash first 4MB for speed
- Return list of AudioFile models
- Log a summary table using rich.table.Table
- Support optional glob pattern filter via config.include_pattern

The converted temp files should go to a subdirectory of cache_dir called `normalized/`.
Keep a record of original_path → normalized_path in the AudioFile model (add `normalized_path: Path | None` if needed).

Important: ffmpeg conversion should use subprocess, not pydub, to avoid loading the full file into memory. Shell out: `ffmpeg -i {input} -ar {sr} -ac {channels} -sample_fmt s16 {output} -y -loglevel error`
```

### Prompt 1.2 — Cache

```
Implement field_miner/cache.py

The cache stores analysis results (list[ScoredWindow]) keyed by (file_hash, config_hash).
Config hash = MD5 of the windowing config fields only (window_size, hop_size, normalize_sr).

Interface:
- `cache.has(audio_file, config) -> bool`
- `cache.load(audio_file, config) -> list[ScoredWindow]`
- `cache.save(audio_file, config, windows: list[ScoredWindow]) -> None`
- `cache.stats() -> dict` — how many files cached, total size

Storage format: one gzipped JSON file per source file, named `{file_hash}_{config_hash}.json.gz`
Use Pydantic's `.model_dump_json()` / `.model_validate_json()` for serialization.
Handle numpy arrays: convert to lists before serializing, restore as np.array on load.
```

### Validation

```bash
# Put a test .wav and a test .mp3 in tests/fixtures/
python -c "
from field_miner.ingestor import Ingestor
from field_miner.models import Config
cfg = Config.from_yaml('config.yaml')
cfg.input_dir = 'tests/fixtures'
ing = Ingestor(cfg)
files = ing.discover()
print(files)
"
```

---

## Session 2 — Analyzer

**Goal**: Extract all features from a windowed audio segment.

**Note**: This is the most complex module. Split it into two prompts — librosa features first, YAMNet second.

### Context block

```
Project: field-miner
Current state: Ingestor and Cache complete and tested.
Today's task: implement field_miner/analyzer.py
The Analyzer slides a window across an AudioFile and returns list[WindowFeatures].
See spec for the full feature table. Models are in field_miner/models.py.
```

### Prompt 2.1 — librosa features

```
Implement the librosa feature extraction portion of field_miner/analyzer.py

Create a class `Analyzer` with method:
  `extract_librosa_features(segment: np.ndarray, sr: int) -> dict`

This method should extract ALL of the "Acoustic Content Features" and "Signal Quality Features" from the spec:

Signal quality:
- peak_amplitude: float
- rms_mean_db: float (convert to dBFS)
- clipping_ratio: float (fraction of samples above 0.98)
- dc_offset: float

Acoustic features:
- spectral_centroid_mean, spectral_centroid_std
- spectral_rolloff_mean, spectral_rolloff_std  
- spectral_contrast (mean per band, returned as list[float])
- spectral_flatness_mean, spectral_flatness_std
- mfcc (13 coefficients, mean and std as separate list[float])
- mfcc_delta_std: float (mean of per-coefficient delta std)
- zcr_mean, zcr_std
- onset_strength_mean, onset_strength_std
- onset_count: int
- rms_variance: float (std of RMS over 8 sub-windows)
- mel_entropy: float (scipy.stats.entropy of mean mel spectrogram)
- harmonic_ratio: float (energy ratio: harmonic / (harmonic + percussive))

All features should handle edge cases: very short segments, all-zero signal, etc. Wrap individual feature computations in try/except and return NaN for failures rather than crashing.

Return a flat dict. Add type hints to the method signature.
```

### Prompt 2.2 — YAMNet integration

```
Now add YAMNet semantic tagging to field_miner/analyzer.py

Add to the Analyzer class:
1. Lazy-load YAMNet from TensorFlow Hub in `__init__` if `use_yamnet=True`:
   `self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')`
   Load class names from the model's `class_map_path()`.

2. Method `extract_yamnet_features(segment: np.ndarray, sr: int) -> dict`:
   - Resample segment to 16kHz mono (use librosa.resample)
   - Normalize to float32 in [-1, 1]
   - Run model: `scores, embeddings, spectrogram = self.yamnet(waveform)`
   - Mean-pool scores across frames → shape (521,)
   - Mean-pool embeddings across frames → shape (1024,) — store as list[float]
   - Return: top_labels: list[tuple[str, float]] (top 5), yamnet_embedding: list[float]

3. Main method `analyze_file(audio_file: AudioFile, config: Config, cache: Cache) -> list[WindowFeatures]`:
   - Load normalized WAV with librosa (sr=config.normalize_sr, mono=True)
   - Slide window with hop, skip windows outside file bounds
   - For each window: call extract_librosa_features, call extract_yamnet_features every N windows (config.yamnet_every_n_windows), carry forward last known YAMNet result for skipped windows
   - Construct and return list[WindowFeatures]
   - Show progress with tqdm, labeled with filename

Important: wrap the entire YAMNet forward pass in a try/except. If TF is unavailable or model fails to load, set use_yamnet=False and return empty labels gracefully. Print a warning via rich.
```

### Gotcha — TensorFlow on Apple Silicon

```
KNOWN ISSUE: If running on Apple Silicon Mac, add this note to the analyzer:
TensorFlow may require tensorflow-metal for GPU, or use tensorflow-macos.
Add a check: if platform.system() == 'Darwin' and platform.machine() == 'arm64':
  print a warning suggesting: pip install tensorflow-macos tensorflow-metal
Fall back to CPU silently.
```

### Validation

```bash
python -c "
import numpy as np
from field_miner.analyzer import Analyzer
from field_miner.models import Config

cfg = Config.from_yaml('config.yaml')
analyzer = Analyzer(cfg, use_yamnet=False)  # test without YAMNet first
fake_audio = np.random.randn(44100 * 20).astype(np.float32)  # 20s of noise
features = analyzer.extract_librosa_features(fake_audio, 44100)
print({k: type(v).__name__ for k, v in features.items()})
print('Feature count:', len(features))
"
```

---

## Session 3 — Scorer and Filter

**Goal**: Implement the composite scoring model and rejection logic.

### Context block

```
Project: field-miner
Current state: Ingestor, Cache, Analyzer complete.
Today's task: implement field_miner/scorer.py and field_miner/filter.py
Both modules operate on WindowFeatures and produce ScoredWindow or rejection decisions.
```

### Prompt 3.1 — Filter

```
Implement field_miner/filter.py

Class `Filter` with config: Config injected at __init__

Method `hard_reject(features: WindowFeatures) -> tuple[bool, str | None]`:
Returns (should_reject, reason_string)

Hard reject conditions (any → reject):
1. peak_amplitude > config.clipping_threshold (default 0.99) → "clipping"
2. clipping_ratio > 0.005 → "clipping_ratio"
3. rms_mean_db < config.silence_threshold_dbfs (default -60) → "silence"
4. abs(dc_offset) > 0.05 → "dc_offset"
5. Any of top-3 YAMNet labels (by score) in config.reject_labels → "rejected_label:{label}"

Method `apply_soft_penalties(features: WindowFeatures, score: float) -> float`:
- If any top-5 label is in reject_labels (but not top-3) → multiply by 0.7
- If any top-5 label in ["Wind noise", "Static", "Hum"] → multiply by 0.6
- If onset_count == 0 AND spectral_centroid_std < 200 → multiply by 0.7 (boring static)
- Score is capped at 1.0 after all penalties

Method `compute_prefer_boost(features: WindowFeatures) -> float`:
- For each label in top-5 that appears in config.prefer_labels → multiply by 1.1
- Cap total boost multiplier at 1.5
- Return the multiplier (caller applies it to score)

Keep a running tally of rejection counts by reason — expose via `filter.rejection_summary() -> dict`.
```

### Prompt 3.2 — Scorer

```
Implement field_miner/scorer.py

Class `Scorer` with config: Config

Core method: `score(features: WindowFeatures) -> ScoredWindow`

The scorer needs percentile normalization, but on first pass it can't know the global distribution. 
Design this in two phases:

Phase 1 — Raw sub-scores (0 to unbounded):
Compute each sub-score as a raw float.

Phase 2 — Normalization:
`Scorer.fit(all_windows: list[WindowFeatures])` — compute per-sub-score percentile stats (5th and 95th percentile) across all windows and store them.
After fit(), `score()` normalizes each raw sub-score using clip((x - p5) / (p95 - p5), 0, 1).

Sub-score implementations:

`_spectral_dynamism(f)` → raw score:
  (spectral_centroid_std / 1000) + (spectral_rolloff_std / 2000)
  Higher variance = more interesting evolution

`_textural_richness(f)` → raw score:
  mean(spectral_contrast) / 40 + mel_entropy / 5
  Rich texture across frequency bands

`_timbral_movement(f)` → raw score:
  mfcc_delta_std (mean across 13 coefficients)
  Timbre changing over time

`_event_interest(f)` → raw score:
  Gaussian curve peaked at onset_count=8 for a 20s window
  score = exp(-0.5 * ((onset_count - 8) / 5) ** 2)
  Zero onsets → near-zero score. >30 onsets → near-zero score.

`_dynamic_range(f)` → raw score:
  rms_variance (already normalized-ish)

`_harmonic_content(f)` → raw score:
  Parabolic curve: peaks at harmonic_ratio ~0.4
  score = 1 - abs(harmonic_ratio - 0.4) / 0.6
  Pure noise or pure tone both score lower than mixed content

Final score = weighted sum using config.weights, then apply prefer_boost from Filter.

`score()` returns ScoredWindow with score and sub_scores dict populated.

Raise ValueError if score() is called before fit() — unless --fast mode bypasses normalization.
```

### Prompt 3.3 — Fast mode

```
Add a `--fast` mode bypass to Scorer:

In fast mode (config.fast = True):
- Skip fit() requirement
- Use hand-tuned scale factors instead of percentile normalization
- Reduces quality slightly but allows streaming/chunked processing without two passes

Add `score_raw(features) -> ScoredWindow` that applies hand-tuned divisors instead of percentile normalization. This is what --fast uses.

Also add a convenience method `Scorer.calibrate_from_sample(windows: list[WindowFeatures])` that runs fit() on a random sample of 500 windows from a larger set — used when the full dataset is too large to hold in memory.
```

### Validation

```bash
python -c "
from field_miner.filter import Filter
from field_miner.models import Config, WindowFeatures

cfg = Config.from_yaml('config.yaml')
f = Filter(cfg)

# Test hard reject: silence
dummy = WindowFeatures(source_file='test.wav', start_s=0, end_s=20,
    peak_amplitude=0.3, rms_mean_db=-65, clipping_ratio=0.0, dc_offset=0.0,
    top_yamnet_labels=[], yamnet_embedding=[],
    # ... fill other fields with 0.0
)
rejected, reason = f.hard_reject(dummy)
assert rejected == True and 'silence' in reason
print('Filter hard reject test: PASS')
"
```

---

## Session 4 — Deduplicator

**Goal**: Prune overlapping and near-duplicate windows before export.

### Context block

```
Project: field-miner
Current state: Ingestor, Cache, Analyzer, Scorer, Filter complete.
Today's task: implement field_miner/deduplicator.py
Input: list[ScoredWindow] from all files. Output: deduplicated list[ScoredWindow].
```

### Prompt 4.1

```
Implement field_miner/deduplicator.py

Class `Deduplicator` with config: Config

Method 1 — `prune_temporal(windows: list[ScoredWindow]) -> list[ScoredWindow]`:
  For windows from the SAME source file only (group by source_file first):
  - Sort by score descending
  - Greedy NMS: keep a window if it doesn't overlap > config.dedup_temporal_overlap 
    with any already-kept window from the same file
  - Overlap = intersection_duration / min(duration_a, duration_b)
  - Return pruned list (cross-file windows are untouched)

Method 2 — `prune_by_embedding(windows: list[ScoredWindow]) -> list[ScoredWindow]`:
  Uses cosine similarity on yamnet_embedding (1024-D float list)
  - Skip windows with empty embeddings (yamnet was disabled)
  - Build numpy matrix of embeddings, compute pairwise cosine similarity
  - Greedy: sort by score descending, keep window if its max cosine similarity 
    to any already-kept window < config.dedup_embedding_similarity
  - For large sets (>1000 windows), use batched approximate comparison:
    only compare against the 50 most recently kept windows
  - Return deduplicated list

Method 3 — `run(windows: list[ScoredWindow]) -> list[ScoredWindow]`:
  Runs prune_temporal first, then prune_by_embedding
  Logs: original count, after temporal, after embedding, final count — using rich

Performance note: for very large window sets, the full pairwise O(n²) cosine matrix
may be too large. Add a guard: if len(windows) > 2000, use the batched approximation
in prune_by_embedding and log a warning that deduplication is approximate.
```

---

## Session 5 — Exporter

**Goal**: Slice audio files and write final clips + metadata.

### Context block

```
Project: field-miner
Current state: All analysis and scoring modules complete.
Today's task: implement field_miner/exporter.py
The Exporter takes list[ScoredWindow], slices original audio files, and writes output.
```

### Prompt 5.1

```
Implement field_miner/exporter.py

Class `Exporter` with config: Config

Main method `export(windows: list[ScoredWindow]) -> list[ExportedClip]`:

Step 1 — Selection:
  - Filter to windows where score >= config.min_score_threshold
  - Sort by score descending
  - Apply per-file cap: max config.max_clips_per_file per source file
  - Apply global cap: max config.max_total_clips total
  - Log selection summary

Step 2 — Audio slicing (per window):
  Method `_slice_clip(window: ScoredWindow) -> Path`:
  - Load the source audio file with soundfile (not librosa — preserves bit depth)
  - Slice the exact sample range: start = int(window.start_s * sr), end = int(window.end_s * sr)
  - Apply fade in: linear ramp over first config.fade_in_ms milliseconds
  - Apply fade out: linear ramp over last config.fade_out_ms milliseconds
  - If config.normalize_output: scale to -6 dBFS peak
  - Write to output_dir/clips/ as 24-bit WAV at original sample rate
  - Filename format: {source_stem}_{start_ms:07d}ms_{score_pct:03d}.wav
    e.g. iceland_river_003_0045200ms_087.wav
  - Return path to written file

Step 3 — Metadata:
  Write results.csv and results.json to output_dir.
  CSV columns: source_file, start_time_s, end_time_s, duration_s, score,
    top_labels (semicolon-joined), spectral_centroid_mean, rms_mean_db,
    onset_count, mfcc_delta_std, harmonic_ratio, clip_path
  JSON: list of ExportedClip.model_dump() with all fields

Step 4 — Score histogram:
  If matplotlib is available, write score_distribution.png to output_dir
  Simple histogram of all candidate scores (pre-threshold), with threshold line marked

Step 5 — Run log:
  Write run_log.txt: processing time, file count, window count, rejection breakdown,
  dedup stats, final clip count

Use tqdm with a description for the slicing loop. Handle missing source files gracefully.
```

### Prompt 5.2 — Ableton extras (optional)

```
Add optional Ableton-friendly export to Exporter.

Method `_write_ableton_sidecar(clip: ExportedClip) -> None`:
  Only runs if config.ableton_export = True (add to Config model)
  
  Write a minimal .asd (Ableton Sample Data) XML sidecar next to each clip.
  .asd is just XML — we're writing a stub that sets clip color.

  Color mapping based on top YAMNet label category:
    water/stream/rain/waves → color id 13 (blue)
    bird/birdsong → color id 9 (green)
    wind/rustling → color id 0 (grey)
    insect/cricket/frog → color id 14 (yellow)
    fire/thunder → color id 4 (orange)
    default → color id 1 (white)

  Minimal .asd XML structure:
  <?xml version="1.0" encoding="UTF-8"?>
  <Ableton MajorVersion="5" MinorVersion="10.0.2" SchemaChangeCount="3" Creator="field-miner" Revision="">
    <SampleData>
      <SampleRef>
        <FileRef><Name>{filename}</Name></FileRef>
      </SampleRef>
      <DefaultDuration>{duration_samples}</DefaultDuration>
      <DefaultSampleRate>{sample_rate}</DefaultSampleRate>
    </SampleData>
  </Ableton>

  Note: full .asd support would require reverse-engineering Ableton's format further.
  This stub is enough for color tagging if opened in Live's browser.
```

---

## Session 6 — CLI Wiring

**Goal**: Connect all modules through the Click CLI and make the tool end-to-end runnable.

### Context block

```
Project: field-miner
Current state: All modules implemented — Ingestor, Cache, Analyzer, Scorer, Filter, Deduplicator, Exporter.
Today's task: wire everything together in field_miner/cli.py and make the full pipeline run.
```

### Prompt 6.1 — CLI

```
Implement field_miner/cli.py with the following Click commands:

@cli.command('analyze')
  Options:
    --input / -i PATH (required)
    --output / -o PATH (required, default ./mined_clips)
    --config / -c PATH (default config.yaml, optional)
    --fast / --no-fast (bypass two-pass scoring normalization)
    --max-clips INT (override config.max_total_clips)
    --no-yamnet (disable YAMNet, faster but no semantic labels)
    --dry-run (report stats without exporting)
    --workers INT (parallel file count)

  Pipeline:
    1. Load config, apply CLI overrides
    2. Print a rich header panel with run settings
    3. Ingestor.discover() → list[AudioFile]
    4. For each file (parallel if workers>1 using concurrent.futures.ProcessPoolExecutor):
       - Check cache, skip if cached
       - Analyzer.analyze_file() → list[WindowFeatures]
       - Filter.hard_reject() each window → keep valid ones
       - Cache.save()
    5. Collect all WindowFeatures from all files
    6. Scorer.fit() on all windows (skip in fast mode)
    7. Score all windows → list[ScoredWindow]
    8. Apply Filter.apply_soft_penalties() and prefer_boost to each
    9. Deduplicator.run() → pruned list
    10. If dry-run: print stats table and exit
    11. Exporter.export() → list[ExportedClip]
    12. Print completion summary with rich

@cli.command('rescore')
  Re-weights an existing results.json without re-analyzing.
  Options: --results PATH, --config PATH, --output PATH
  Loads all ScoredWindow objects from JSON, re-runs Scorer with new weights, re-exports.

@cli.command('stats')
  Options: --results PATH
  Prints a rich table of: total clips, score distribution (min/median/max/p90),
  top 10 labels, per-file breakdown, rejection summary.

@cli.command('review')
  Options: --results PATH, --top INT (default 20)
  Prints the top N clips as a rich table (path, score, labels, duration).
  If --play flag and mpv/afplay available, offer to play each clip interactively.

Add a rich startup banner to the main cli group showing the tool name.
```

### Prompt 6.2 — Parallel processing

```
The analyze command's parallel processing needs care.

The main issue: TensorFlow (YAMNet) doesn't fork safely in multiprocessing.
Solution: use a process pool only for librosa feature extraction, run YAMNet in 
the main process as a second pass.

Update the analyze pipeline:
  Phase A (parallel): For each file, extract librosa features only → returns list[WindowFeatures] with empty yamnet fields
  Phase B (main process, sequential): For files not in cache, run YAMNet on their windows in batches
  Phase C: Merge features, cache, continue to scoring

Add a note in the code explaining why YAMNet must run in main process.

Also add: if workers=1, skip ProcessPoolExecutor entirely and use a simple loop.
This avoids pickling issues during development/debugging.
```

---

## Session 7 — Tests

**Goal**: Build a test suite that validates each module without requiring real audio files.

### Context block

```
Project: field-miner
Current state: Full pipeline implemented. 
Today's task: implement the test suite using pytest and synthetic audio fixtures.
```

### Prompt 7.1 — Fixtures

```
Create tests/conftest.py and tests/fixtures/ with synthetic audio fixtures.

In conftest.py, create pytest fixtures:

@pytest.fixture
def silence_array(): 
  return np.zeros(44100 * 20, dtype=np.float32), 44100  # 20s silence

@pytest.fixture
def clipped_array():
  y = np.ones(44100 * 20, dtype=np.float32) * 1.01  # clipped sine
  return y, 44100

@pytest.fixture 
def noise_array():
  return np.random.randn(44100 * 20).astype(np.float32), 44100

@pytest.fixture
def chirp_array():
  # Frequency sweep: interesting spectral dynamism
  import scipy.signal
  t = np.linspace(0, 20, 44100 * 20)
  y = scipy.signal.chirp(t, f0=200, f1=4000, t1=20, method='logarithmic').astype(np.float32)
  return y, 44100

@pytest.fixture
def default_config():
  return Config(input_dir='/tmp', output_dir='/tmp/out', cache_dir='/tmp/.cache')

Also create a small real WAV file at tests/fixtures/test_20s.wav:
  20 seconds of pink noise + a few sine bursts — use scipy to generate and soundfile to write.
  Add this generation to a conftest setup function.
```

### Prompt 7.2 — Module tests

```
Implement tests/test_filter.py and tests/test_scorer.py

test_filter.py:
  test_hard_reject_silence: silence_array → should reject with reason 'silence'
  test_hard_reject_clipping: clipped_array → should reject with reason 'clipping'
  test_hard_reject_speech_label: WindowFeatures with top label 'Speech' → should reject
  test_no_reject_noise: noise_array at normal level → should not reject
  test_prefer_boost_birds: WindowFeatures with top label 'Bird' → boost > 1.0
  test_wind_penalty: WindowFeatures with label 'Wind noise' in top 5 → score reduced

test_scorer.py:
  test_score_range: all sub-scores are [0, 1] after fit()
  test_chirp_beats_silence: chirp_array should score higher than silence_array
  test_noise_beats_silence: noise_array should score higher than silence_array
  test_rescore_consistency: same input → same score
  test_fast_mode_no_fit_required: Scorer(config, fast=True).score() works without fit()

Use mock WindowFeatures with hand-crafted feature values to test edge cases without needing the Analyzer.
```

### Validation

```bash
pytest tests/ -v --tb=short
# Target: all tests pass, no import errors
```

---

## Session 8 — Polish and Hardening

**Goal**: Error handling, progress resumption, edge cases.

### Prompt 8.1 — Hardening pass

```
Do a hardening pass across the entire codebase. For each module, check for and fix:

1. field_miner/ingestor.py:
   - What happens if ffmpeg is not installed? Raise a clear error with install instructions.
   - What if a file is corrupt and can't be read? Log warning and skip, don't crash.
   - What if input_dir doesn't exist? Raise FileNotFoundError immediately with message.

2. field_miner/analyzer.py:
   - What if a window is shorter than 1 second? Skip it, log debug.
   - What if librosa raises an exception on a window? Log warning with timestamp, 
     return WindowFeatures with all features = 0.0 and a flag `analysis_failed=True`.
   - What if YAMNet download fails (no internet)? Fall back to no-yamnet mode gracefully.

3. field_miner/exporter.py:
   - What if output_dir already exists with clips in it? Don't overwrite — append a 
     timestamp suffix to the run's subdirectory, or prompt user.
   - What if a source audio file has been moved/deleted since analysis? Log error, 
     mark clip as export_failed in metadata.

4. General:
   - Add a top-level exception handler in cli.py that catches unexpected errors,
     prints a friendly rich error panel, and suggests running with --debug flag.
   - Add --debug flag that enables Python traceback on error.
```

### Prompt 8.2 — Resume support

```
Add run resume support to the analyze command.

The idea: if a run crashes halfway through, re-running with the same --output dir 
should pick up where it left off.

Implementation:
1. On start, check if output_dir/run_manifest.json exists
2. If it does, load the manifest and identify which AudioFiles were already fully analyzed 
   (have a complete cache entry)
3. Skip those files in the ingestor discovery loop
4. Print: "Resuming previous run: N files already complete, M remaining"
5. After all files processed, merge new results with existing results.json if present

Add --resume flag (default True) and --no-resume to force a full restart.
```

---

## Session 9 — README and Final Docs

### Prompt 9.1

```
Write a complete README.md for field-miner.

Include:
1. One-paragraph description (mention: IDM/ambient music production, field recordings, automatic mining)
2. Install section (pip, ffmpeg requirement, Apple Silicon note)
3. Quick start (5-line example: install → put recordings in folder → run → check output)
4. Full CLI reference (all commands and options as a table)
5. config.yaml annotated reference (every field explained)
6. How scoring works (plain English, one paragraph per sub-score)
7. Label filtering: how to customize reject_labels and prefer_labels for your use case
8. Performance tips: --fast mode, --workers, cache behavior
9. Output structure: what's in the clips/ folder, what results.csv contains
10. Troubleshooting: TF issues, ffmpeg not found, Apple Silicon

Keep it technical but practical. This is a tool for a musician-developer, not a data scientist.
```

---

## Full Integration Test

Once all sessions are complete, run this full end-to-end test before calling it done:

```bash
# Generate a 5-minute synthetic "field recording" for testing
python - <<'EOF'
import numpy as np
import soundfile as sf
import os

os.makedirs('test_recordings', exist_ok=True)
sr = 44100
duration = 300  # 5 minutes

t = np.linspace(0, duration, sr * duration)
# Pink noise base
noise = np.random.randn(sr * duration).astype(np.float32) * 0.1
# Add some "events" — sine bursts
for event_t in [30, 90, 120, 180, 240]:
    start = int(event_t * sr)
    end = int((event_t + 3) * sr)
    freq = np.random.choice([440, 880, 1200, 2400])
    noise[start:end] += 0.3 * np.sin(2 * np.pi * freq * t[start:end]).astype(np.float32)

sf.write('test_recordings/synthetic_field_001.wav', noise, sr)
print("Written 5-minute test recording.")
EOF

# Run the full pipeline
field-miner analyze \
  --input ./test_recordings \
  --output ./test_output \
  --no-yamnet \
  --fast \
  --max-clips 20

# Check output
field-miner stats --results ./test_output/results.json
ls -la ./test_output/clips/
```

Expected output: 10–20 WAV clips in `test_output/clips/`, `results.csv` with scores, `score_distribution.png`.

---

## Agentic Harness Tips

Working with Claude Code on this project — things that help:

**Commit between sessions.** Each session in this guide produces a testable module. Commit it before the next session. This gives Claude Code clean git context and lets you `git diff` to review changes.

**Feed the spec back.** At the start of each session, paste the relevant section of the spec as a reminder. Claude Code's context window is large but the spec anchors decisions.

**Use `--allowed-tools` wisely.** For implementation sessions, allow `Bash`, `Edit`, `Write`. For the hardening session, add `Read` so it can audit the whole codebase without asking.

**The rescore command is your iteration loop.** Once you have real recordings processed, you'll tune weights constantly. Running `field-miner rescore` with different `config.yaml` files is faster than thinking about what to prompt.

**Validation after every session.** Don't skip the validation blocks. A broken module two sessions back is much harder to debug than one found immediately.

**If Claude Code diverges from the spec**, paste the relevant spec section and say: *"The spec says X. Your implementation does Y. Please reconcile, keeping the spec's intent."*
