# Field Recording Miner — System Spec

## Overview

A CLI tool that ingests large libraries of location-specific field recordings and automatically mines them for musically interesting snippets suitable for ambient/IDM music production. The system scores every windowed segment of every file against a composite "interestingness" model, rejects undesirable content, deduplicates similar results, and exports sliced audio with rich metadata.

---

## Goals

- Process GBs of long field recordings with zero manual pre-listening
- Surface 10–30s snippets (up to 60s for loopable ambient beds) that are texturally, spectrally, or dynamically interesting
- Reject bad audio (clipping, wind noise, speech, sirens, silence)
- Tag each result semantically (birds, water, insects, wind, etc.)
- Produce a reviewable output folder + metadata CSV/JSON for DAW import
- Be fully re-runnable, resumable, and parameterizable

---

## Tech Stack

- **Python 3.11+**
- `librosa` — core feature extraction and segmentation
- `soundfile` / `pydub` — audio I/O and slicing
- `tensorflow` + `tensorflow_hub` — YAMNet for semantic tagging
- `numpy`, `scipy` — signal math
- `ffmpeg` (system) — format conversion (AIFF, WAV, FLAC, MP3 support)
- `tqdm` — progress bars
- `rich` — CLI output
- `click` — CLI interface
- `pandas` — results aggregation and CSV export
- Optional: `pedalboard` (Spotify) — for post-processing exported clips (fade in/out, normalize)

---

## Architecture

```
input_dir/
  ├── recording_001.wav
  ├── recording_002.aif
  └── ...

CLI → Ingestor → Analyzer → Scorer → Filter → Deduplicator → Exporter
                                                     ↓
                                             output_dir/
                                               ├── clips/
                                               │   ├── [source]_[start]s_[score].wav
                                               │   └── ...
                                               ├── results.csv
                                               └── results.json
```

---

## Module Specs

### 1. Ingestor (`ingestor.py`)

**Responsibility**: Discover, validate, and normalize all audio files.

- Recursively walk `input_dir` for `.wav`, `.aif`, `.aiff`, `.flac`, `.mp3`
- Convert non-WAV formats to temporary 16-bit 44.1kHz WAV via ffmpeg (keep originals)
- Skip files shorter than `min_file_duration` (default: 30s)
- Log file inventory to `run_manifest.json` (path, duration, sample rate, channels, size)
- Support `--include-pattern` glob filter (e.g., `iceland_*`)
- Support resume: skip files already present in manifest

**Config params**:
```yaml
input_dir: str
min_file_duration: 30  # seconds
normalize_sr: 44100
normalize_channels: 1  # mono downmix
```

---

### 2. Analyzer (`analyzer.py`)

**Responsibility**: Slide a window across each file and extract a feature vector per window.

**Windowing strategy**:
- Window size: `window_size` seconds (default: 20s)
- Hop size: `hop_size` seconds (default: 5s) — 75% overlap is fine, gives dense coverage
- Each window → feature dict

**Feature extraction per window** (all via `librosa` unless noted):

#### Signal Quality Features (used for rejection, not scoring)
| Feature | Method | Purpose |
|---|---|---|
| Peak amplitude | `np.max(np.abs(y))` | Detect clipping (>0.99) |
| RMS energy | `librosa.feature.rms` | Detect silence |
| DC offset | `np.mean(y)` | Quality check |
| Clipping ratio | fraction of samples >0.98 | Hard reject |

#### Acoustic Content Features (used for scoring)
| Feature | Method | Notes |
|---|---|---|
| Spectral centroid (mean, std) | `librosa.feature.spectral_centroid` | High std = evolving brightness |
| Spectral rolloff (mean, std) | `librosa.feature.spectral_rolloff` | Frequency envelope dynamics |
| Spectral contrast (per-band) | `librosa.feature.spectral_contrast` | Textural richness |
| Spectral flatness (mean, std) | `librosa.feature.spectral_flatness` | Tonal vs noisy character |
| MFCC (13 coeff, mean + std) | `librosa.feature.mfcc` | Timbre fingerprint |
| MFCC delta std | `librosa.feature.delta(mfcc)` | Timbre change over time |
| Zero-crossing rate (mean, std) | `librosa.feature.zero_crossing_rate` | Texture indicator |
| Onset strength envelope | `librosa.onset.onset_strength` | Event density |
| Onset count | `librosa.onset.onset_detect` | Number of discrete events |
| RMS variance | std of RMS over sub-windows | Dynamic movement |
| Mel spectrogram entropy | `scipy.stats.entropy` on mel bins | Spectral complexity |
| Harmonic ratio | `librosa.effects.hpss` | Proportion harmonic vs percussive |

#### Semantic Features (YAMNet)
- Downsample window to 16kHz mono for YAMNet input
- Run YAMNet, get 521-class probability scores averaged over frames
- Extract top-5 labels + scores
- Map to semantic buckets (see Semantic Buckets section)

**Performance note**: YAMNet is the bottleneck. Run on every Nth window (default: every 3rd), interpolate labels for skipped windows. Cache embeddings to disk (`.cache/` dir keyed by file hash + timestamp).

---

### 3. Scorer (`scorer.py`)

**Responsibility**: Convert feature vectors into a single composite interestingness score [0.0–1.0].

The score is a weighted sum of sub-scores. All weights are configurable via `config.yaml`.

#### Sub-scores

**Spectral Dynamism** (weight: 0.25)
- High std of spectral centroid over time → high score
- High std of spectral rolloff → bonus
- Captures "does the frequency character evolve?"

**Textural Richness** (weight: 0.20)
- High spectral contrast across bands → rich texture
- Mel spectrogram entropy → complex frequency distribution
- Captures "is there interesting stuff happening across the spectrum?"

**Timbral Movement** (weight: 0.20)
- High MFCC delta std → timbre is changing
- Captures "does it feel static or alive?"

**Event Interest** (weight: 0.15)
- Onset density score: neither empty (0 onsets) nor too dense (percussive chaos)
- Sweet spot: 2–15 onsets per 20s window → max score
- Captures "is there something happening without being overwhelming?"

**Dynamic Range** (weight: 0.10)
- High RMS variance within window → interesting amplitude movement
- Not dead-flat, not erratically spiky

**Harmonic Content** (weight: 0.10)
- Some harmonic presence (birds, resonance, tonal textures) → bonus
- Too much harmonic (sounds like music) → slight penalty

**Loopability Bonus** (optional, weight: 0.0 or +0.1)
- Compare start and end 0.5s MFCC fingerprints
- Low difference → potentially loopable → bonus applied only when `--score-loops` flag used

#### Score normalization
- All sub-scores normalized [0,1] using percentile-based min/max across the full run
- Final score = weighted sum, clipped to [0.0, 1.0]

---

### 4. Filter (`filter.py`)

**Responsibility**: Hard-reject windows that fail quality or content criteria.

#### Hard Reject Conditions (any = skip)
- `peak_amplitude > 0.99` (clipping)
- `clipping_ratio > 0.005` (>0.5% samples clipped)
- `rms_mean < silence_threshold` (configurable, default: -60 dBFS)
- `dc_offset > 0.05`
- YAMNet top-3 contains any label from `reject_labels` set

#### Soft Reject (lower score but keep)
- YAMNet detects wind noise → multiply score by 0.6
- Very low onset count with flat spectral centroid → uninspiring silence/drone → multiply by 0.7

#### Default `reject_labels` (curated list, overridable in config)
```yaml
reject_labels:
  # Human sounds
  - Speech
  - Conversation
  - Narration, monologue
  - Singing
  - Screaming
  - Child speech
  # Urban noise
  - Traffic noise
  - Car
  - Motorcycle
  - Siren
  - Air horn
  - Truck
  # Bad signal
  - Static
  - White noise  # only if dominant
  - Hum
```

#### Default `prefer_labels` (used to boost score)
```yaml
prefer_labels:
  - Bird
  - Birdsong
  - Water
  - Stream
  - Rain
  - Wind
  - Rustling leaves
  - Insect
  - Cricket
  - Frog
  - Waves, surf
  - Fire
  - Thunder
  - Waterfall
  - Forest
  - Ambient music  # interesting if incidental
```

Each preferred label present in top-5 → multiply score by 1.1 (capped at 1.5x total boost).

---

### 5. Deduplicator (`deduplicator.py`)

**Responsibility**: Prevent exporting highly similar or heavily overlapping segments.

**Strategy 1 — Temporal overlap pruning**:
- For windows from the same source file, if two windows overlap by >50% AND both score above threshold, keep only the higher-scoring one (non-maximum suppression style)

**Strategy 2 — Embedding similarity**:
- Use YAMNet 1024-D embeddings (mean-pooled per window) as audio fingerprints
- Compute cosine similarity between all candidate windows (across files too)
- If similarity > `dedup_threshold` (default: 0.92), keep only the top-scoring instance
- This catches the same ambient scene recorded multiple times

---

### 6. Exporter (`exporter.py`)

**Responsibility**: Slice and export final clips.

- Export top-N clips per source file (default: `max_clips_per_file: 10`)
- Global export cap: `max_total_clips: 500` (configurable)
- Export only clips scoring above `min_score_threshold` (default: 0.5)

**Clip output**:
- Format: 24-bit WAV, original sample rate (no resampling of output)
- Apply 50ms fade-in and 100ms fade-out (via `pedalboard` or manual numpy)
- Filename: `{source_stem}_{start_time_ms:07d}ms_{score_pct:03d}.wav`
  - e.g., `iceland_river_003_0045200ms_087.wav`
- Optionally normalize to -6 dBFS peak (flag: `--normalize-output`)

**Metadata export** (`results.csv` + `results.json`):
```
source_file, start_time_s, end_time_s, duration_s, score,
top_labels, spectral_centroid_mean, rms_mean, onset_count,
mfcc_delta_std, harmonic_ratio, loopability_score, clip_path
```

**Ableton-friendly extras** (optional flag `--ableton`):
- Generate an Ableton Live-compatible Info.xmp sidecar (sets clip color by top label category)
- Color scheme: water=blue, birds=green, wind=grey, insects=yellow, fire=orange, other=white

---

## CLI Interface

```bash
# Basic run
field-miner analyze \
  --input /path/to/recordings \
  --output /path/to/output \
  --config config.yaml

# Quick preview run (faster, lower quality, top 50 clips only)
field-miner analyze --input ./recs --output ./out --fast --max-clips 50

# Re-score with different weights (no re-analysis)
field-miner rescore \
  --results ./out/results.json \
  --config new_weights.yaml \
  --output ./out_v2

# Show stats on a completed run
field-miner stats --results ./out/results.json

# Play top clips interactively (requires `mpv` or `afplay`)
field-miner review --results ./out/results.json --top 20
```

---

## Config File (`config.yaml`)

```yaml
# Input/output
input_dir: ./recordings
output_dir: ./mined_clips
cache_dir: ./.cache

# Windowing
window_size: 20        # seconds
hop_size: 5            # seconds
min_clip_duration: 10  # floor for export
max_clip_duration: 60  # ceiling for export

# Scoring weights (must sum to 1.0)
weights:
  spectral_dynamism: 0.25
  textural_richness: 0.20
  timbral_movement: 0.20
  event_interest: 0.15
  dynamic_range: 0.10
  harmonic_content: 0.10

# Filtering
silence_threshold_dbfs: -60
clipping_threshold: 0.99
reject_labels: [see above]
prefer_labels: [see above]
wind_noise_penalty: 0.6

# Deduplication
dedup_temporal_overlap: 0.5    # 50% overlap → deduplicate
dedup_embedding_similarity: 0.92

# Export
max_clips_per_file: 10
max_total_clips: 500
min_score_threshold: 0.50
normalize_output: false
fade_in_ms: 50
fade_out_ms: 100
score_loops: false

# Performance
yamnet_every_n_windows: 3
n_workers: 4              # parallel file processing
```

---

## Processing Pipeline Pseudocode

```python
for audio_file in ingestor.discover():
    if cache.has(audio_file):
        windows = cache.load(audio_file)
    else:
        y, sr = load_audio(audio_file, sr=44100, mono=True)
        windows = []
        for start, end in sliding_window(y, sr, window_size, hop_size):
            segment = y[start:end]
            features = analyzer.extract(segment, sr)
            if filter.hard_reject(features):
                continue
            features['score'] = scorer.score(features)
            features['score'] = filter.apply_soft_penalties(features)
            windows.append(Window(start_s, end_s, features))
        cache.save(audio_file, windows)
    
    candidates = deduplicator.prune_temporal(windows)
    all_candidates.extend(candidates)

all_candidates = deduplicator.prune_by_embedding(all_candidates)
all_candidates.sort(by='score', descending=True)
top_clips = select_top(all_candidates, per_file_cap, global_cap, min_score)
exporter.export(top_clips)
```

---

## Output Structure

```
output_dir/
  clips/
    iceland_river_001_0012500ms_091.wav
    iceland_wind_003_0087200ms_083.wav
    ...
  results.csv          ← full metadata table
  results.json         ← same, JSON format
  run_manifest.json    ← what files were processed
  run_log.txt          ← warnings, rejected counts, timings
  score_distribution.png  ← histogram of scores (matplotlib)
```

---

## Performance Targets

| Dataset size | Target runtime |
|---|---|
| 1 GB | < 5 min |
| 10 GB | < 45 min |
| 50 GB | < 4 hrs |

Achieved via: parallel file processing (`multiprocessing`), embedding cache (skip YAMNet on re-runs), numpy vectorized feature extraction.

---

## Future / Optional Enhancements

- **Interactive web review UI**: HTML page with audio players, score filter slider, label filter checkboxes — served locally via `flask` or pure static HTML from results JSON
- **Loop detection pass**: Second pass on top candidates specifically scoring loopability (start/end spectral similarity, zero-crossing alignment)
- **Similarity clustering**: UMAP on YAMNet embeddings → cluster visualization to explore what you have
- **Custom reject model**: Fine-tune a small classifier on user-labeled good/bad clips to personalize scoring
- **Ableton Pack export**: Auto-generate an `.alp` folder structure with categorized clips ready to drop in
- **GPS/timestamp metadata**: If recordings have embedded GPS (from field recorder), attach location data to results

---

## Notes for Implementation

- Test suite should include a `tests/fixtures/` dir with short synthetic audio files (pure tones, noise, silence, clipping) to validate each filter
- The `rescore` command is important — analyzing GBs of audio takes time, but re-weighting a finished feature cache should be near-instant
- Consider a `--dry-run` flag that reports stats (how many files, estimated runtime, clip count at current thresholds) without exporting
- YAMNet requires 16kHz input — always downsample a copy for inference, export at original quality
