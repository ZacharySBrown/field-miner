# field-miner

A CLI tool that ingests large libraries of field recordings and automatically mines them for musically interesting snippets suitable for ambient/IDM music production. Scores every windowed segment against a composite "interestingness" model, rejects bad audio, deduplicates, and exports sliced audio with rich metadata.

## Quick start

### macOS

```bash
# Clone
git clone git@github.com:zacharysbrown/field-miner.git
cd field-miner

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (core)
pip install -e .

# Install with YAMNet semantic tagging (optional, adds TensorFlow)
pip install -e ".[yamnet]"

# Install everything (YAMNet + pedalboard + dev tools)
pip install -e ".[all]"

# System dependency
brew install ffmpeg

# Run
field-miner mine /path/to/field-recordings --output ./output
```

### Windows 10/11 (via WSL2)

1. **Enable WSL2** -- open PowerShell as Administrator:
   ```powershell
   wsl --install
   ```
   Reboot when prompted. Complete Ubuntu setup. Then:
   ```powershell
   wsl --set-default-version 2
   ```

2. **Install system dependencies** in WSL terminal:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg libsndfile1
   ```

3. **Clone into the WSL filesystem** (not `/mnt/c/...` -- audio I/O is dramatically slower on the Windows mount):
   ```bash
   cd ~
   git clone git@github.com:zacharysbrown/field-miner.git
   cd field-miner
   ```

4. **Create virtual environment and install**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all]"
   ```

5. **Copy recordings into WSL** for best performance:
   ```bash
   # Option A: copy from Windows
   cp -r /mnt/c/Users/$(whoami)/Music/FieldRecordings ~/recordings

   # Option B: work directly from Windows mount (slower but no copy)
   field-miner mine /mnt/c/Users/$(whoami)/Music/FieldRecordings --output ~/output
   ```

6. **Run**:
   ```bash
   field-miner mine ~/recordings --output ~/output
   ```

### Linux

```bash
sudo apt install python3.11 python3.11-venv ffmpeg libsndfile1
git clone git@github.com:zacharysbrown/field-miner.git
cd field-miner
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
field-miner mine /path/to/recordings --output ./output
```

## What it does

- Processes GBs of long field recordings with zero manual pre-listening
- Surfaces 10-30s snippets (up to 60s for loopable ambient beds) that are texturally, spectrally, or dynamically interesting
- Rejects bad audio (clipping, wind noise, speech, sirens, silence)
- Tags each result semantically via YAMNet (birds, water, insects, wind, etc.)
- Produces a reviewable output folder + metadata CSV/JSON for DAW import
- Fully re-runnable, resumable, and parameterizable

## Configuration

Edit `config.yaml` to tune extraction parameters:

```yaml
# See config.yaml for full reference
mining:
  window_sec: 15
  hop_sec: 5
  min_score: 0.4
  max_results: 50
```

## Reference

- [System spec](field-recording-miner-spec.md)
- [Claude Code guide](field-recording-miner-claude-code-guide.md)
