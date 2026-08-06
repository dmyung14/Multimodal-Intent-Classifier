# Milestone 2 Real MIntRec Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Milestone 2's toy synthetic multimodal pipeline with real MIntRec audio/video data and real frozen pretrained encoders (wav2vec2-base for audio, ResNet18 for video), completing `CLAUDE.md`'s M2 goal.

**Architecture:** A three-stage pipeline, each stage a separate script producing files the next stage reads: (1) download real clips + extract raw audio/frames via `ffmpeg`, (2) run both encoders once per clip and cache embeddings (avoids re-running expensive neural inference on every training iteration), (3) load cached embeddings, fuse via concatenation + `StandardScaler`, train/evaluate `LogisticRegression` — same shape as the toy pipeline, real data underneath. A fourth stage narrates it all in a notebook.

**Tech Stack:** Python 3.11 in the `mmi` conda environment. `pandas`/`numpy`/`scikit-learn`/`matplotlib` (already installed), `huggingface_hub`/`torch`/`transformers` (already installed from M1), `ffmpeg` (already installed per `environment.yml`). New this plan: `torchvision` (small install) for `ResNet18`.

## Global Constraints

- New files only — `src/generate_toy_multimodal.py` and `src/train_multimodal_toy.py` stay untouched (permanent, zero-download sanity check).
- Follow established code patterns from `src/train_multimodal_toy.py` and `src/train_mintrec.py`: module docstring (goal / how to run / terms), `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, `"=" * 70` banners, `load_data()`/`split_data()`/`evaluate()` naming where applicable, `evaluate()` writing predictions to a `results/*.csv` file.
- Same stratified `train_test_split` as every prior script: `test_size=0.20` then `test_size=0.1875`, both `random_state=42`, stratified on the label column.
- No pytest suite exists or should be introduced — verification is "run the script, read the printed output, check the generated files," per this project's established precedent.
- **Real accuracy must be reported honestly.** Unlike the toy increment's engineered 0.700, this pipeline's accuracy against Milestone 0's 0.550 baseline is genuinely unknown — do not adjust methodology to chase a target number.
- **Downloads/installs need explicit human confirmation before they happen** — this project's `CLAUDE.md` requires confirming before any download over ~100MB or any package install, and this plan crosses that line twice (Task 1: ~330MB data download; Task 2: `torchvision` install + ~405MB combined model weights). Each task below has a bolded pre-dispatch note — **the controller running this plan must get that confirmation from the human before dispatching the task's implementer**, not delegate the asking to the implementer itself.
- All new data lives under `data/mintrec_multimodal/`, gitignored (matches the existing `.gitignore`'s `data/*` pattern — no `.gitignore` change needed).
- The project's `commit` skill (secret scan, draft message, staged-files review) is used for every commit, not raw `git commit`.

---

## Task 1: Acquire real clips, extract raw audio/frames

> **Before dispatching this task: confirm with the human that it's OK to download the default 300-clip stratified subset (~330MB) from the `THU-IAR/MIntRec` Hugging Face mirror.** Do not proceed without that confirmation — this is exactly the kind of download `CLAUDE.md` requires asking about first.

**Files:**
- Create: `src/generate_mintrec_multimodal.py`

**Interfaces:**
- Consumes: `data/mintrec/all.tsv` (existing, from M1 — columns `season, episode, clip, text, label`).
- Produces: `data/mintrec_multimodal/index.csv` with columns `sample_id, text, intent, season, episode, clip, audio_path, frame_dir`. `sample_id` is `{season}_{episode}_{clip}` (e.g. `S04_E01_103`).
- Produces: `data/mintrec_multimodal/raw_clips/{season}/{episode}/{clip}.mp4`, `data/mintrec_multimodal/audio/{sample_id}.wav` (mono, 16kHz, 16-bit PCM), `data/mintrec_multimodal/frames/{sample_id}/frame_{0..4}.jpg` (5 JPEGs).
- Constants later tasks depend on: `AUDIO_SAMPLE_RATE = 16000` (16-bit PCM WAV, mono) — Task 2's audio loading code reads this format directly via Python's stdlib `wave` module, no other audio library needed.

- [ ] **Step 1: Write `src/generate_mintrec_multimodal.py`**

```python
"""
Milestone 2, increment 2: acquire real MIntRec video/audio and extract raw
audio/frame files for the real frozen-encoder pipeline.
=================================================================
GOAL OF THIS FILE
    Milestone 2's toy increment (src/generate_toy_multimodal.py) proved the
    fusion pipeline's mechanics on fully synthetic, invented data. This
    file replaces that synthetic data with the REAL thing: real short
    video clips from the MIntRec dataset (dialogue from the Superstore TV
    show), downloaded from the THU-IAR/MIntRec mirror on Hugging Face, with
    real audio tracks and real video frames extracted from each clip via
    ffmpeg.

    src/generate_toy_multimodal.py and src/train_multimodal_toy.py are left
    untouched -- they remain a permanent, zero-download, instant sanity
    check. This is a new, separate pipeline for the real data.

HOW TO RUN
    conda activate mmi
    python src/generate_mintrec_multimodal.py

    Default run downloads a STRATIFIED SUBSET: 15 clips per intent class
    (300 clips total, ~330MB). This is a real download over this project's
    "confirm before >100MB" rule -- confirm with Claude Code before running
    this the first time.

    python src/generate_mintrec_multimodal.py --full

    Downloads all 2,213 available clips (~2.3GB) instead of the subset --
    a much bigger download, needing its own separate confirmation.

    Safe to re-run: already-downloaded clips are skipped, not re-fetched.

TERMS YOU'LL SEE
    - stratified sample : picking N examples from EACH class, rather than N
                           examples overall, so rare classes aren't left out
    - ffmpeg / ffprobe   : command-line tools for working with audio/video
                           files. ffprobe reads a file's metadata (like its
                           duration); ffmpeg converts or extracts from it.
    - WAV                : an uncompressed audio file format. The real audio
                           encoder used in the next script (wav2vec2) expects
                           a specific sample rate (16kHz) and channel count
                           (mono, 1 channel) -- we convert to that here so
                           nothing needs resampling later.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_TSV_PATH = PROJECT_ROOT / "data" / "mintrec" / "all.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal"

REPO_ID = "THU-IAR/MIntRec"
SAMPLES_PER_CLASS = 15
SEED = 0
AUDIO_SAMPLE_RATE = 16000  # wav2vec2's expected input rate
N_FRAMES = 5

# (season, episode, clip) rows present in all.tsv with NO matching file in
# the THU-IAR/MIntRec Hugging Face mirror's raw_data/ folder -- verified
# directly against the mirror's file listing (2,213 files vs. all.tsv's
# 2,224 rows). Skipped up front so we never attempt a doomed download.
KNOWN_MISSING = {
    ("S05", "E07", 96),
    ("S05", "E15", 83),
    ("S05", "E15", 82),
    ("S05", "E07", 82),
    ("S05", "E15", 86),
    ("S05", "E09", 91),
    ("S05", "E07", 87),
    ("S05", "E15", 85),
    ("S05", "E15", 94),
    ("S05", "E15", 8),
    ("S05", "E09", 90),
}


def load_available_rows():
    """Read all.tsv and drop the 11 rows known to be missing from the mirror."""
    df = pd.read_csv(ALL_TSV_PATH, sep="\t")
    is_missing = df.apply(lambda r: (r["season"], r["episode"], r["clip"]) in KNOWN_MISSING, axis=1)
    available = df[~is_missing].reset_index(drop=True)
    print(f"Loaded {len(df)} rows from all.tsv; {is_missing.sum()} are known-missing from the")
    print(f"Hugging Face mirror and are excluded, leaving {len(available)} available rows.")
    return available


def sample_stratified(df, n_per_class, seed):
    """Take exactly n_per_class rows from every intent, so no class is left out."""
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def download_clip(season, episode, clip):
    """Download one raw clip from the Hugging Face mirror, skipping if already local."""
    local_path = OUTPUT_DIR / "raw_clips" / season / episode / f"{clip}.mp4"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"raw_data/{season}/{episode}/{clip}.mp4",
        )
    except Exception as e:
        raise SystemExit(
            f"\nFailed to download raw_data/{season}/{episode}/{clip}.mp4 from {REPO_ID}: {e}\n"
            "This clip was expected to exist -- check your network connection and try again."
        )
    shutil.copy2(cached_path, local_path)
    return local_path


def extract_audio(clip_path, sample_id):
    """ffmpeg: pull the audio track out as mono 16kHz 16-bit PCM WAV."""
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{sample_id}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-vn",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ],
        check=True, capture_output=True,
    )
    return audio_path


def get_duration_seconds(clip_path):
    """ffprobe: read the clip's duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(clip_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_frames(clip_path, sample_id):
    """ffmpeg: pull N_FRAMES evenly-spaced JPEG frames (midpoints of N equal segments)."""
    frame_dir = OUTPUT_DIR / "frames" / sample_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_seconds(clip_path)
    for i in range(N_FRAMES):
        timestamp = duration * (i + 0.5) / N_FRAMES
        frame_path = frame_dir / f"frame_{i}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(clip_path),
                "-frames:v", "1", str(frame_path),
            ],
            check=True, capture_output=True,
        )
    return frame_dir


def main():
    parser = argparse.ArgumentParser(description="Acquire real MIntRec audio/video and extract raw audio/frames")
    parser.add_argument("--full", action="store_true", help="download all 2,213 available clips instead of a 300-clip subset")
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE REAL MINTREC MULTIMODAL DATA")
    print("=" * 70)

    available = load_available_rows()
    if args.full:
        selected = available
        print(f"\n--full: using all {len(selected)} available rows.")
    else:
        selected = sample_stratified(available, SAMPLES_PER_CLASS, SEED)
        print(f"\nStratified sample: {SAMPLES_PER_CLASS} rows/class -> {len(selected)} rows.")

    rows = []
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        sample_id = f"{row.season}_{row.episode}_{row.clip}"
        print(f"[{i}/{len(selected)}] {sample_id} ({row.label})")

        clip_path = download_clip(row.season, row.episode, row.clip)
        audio_path = extract_audio(clip_path, sample_id)
        frame_dir = extract_frames(clip_path, sample_id)

        rows.append(
            {
                "sample_id": sample_id,
                "text": row.text,
                "intent": row.label,
                "season": row.season,
                "episode": row.episode,
                "clip": row.clip,
                "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                "frame_dir": frame_dir.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    index_df = pd.DataFrame(rows)
    index_path = OUTPUT_DIR / "index.csv"
    index_df.to_csv(index_path, index=False)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote {len(index_df)} rows to {index_path.relative_to(PROJECT_ROOT)}")
    print("Next: python src/extract_mintrec_embeddings.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after confirming the ~330MB download with the human)**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/generate_mintrec_multimodal.py
```

Expected: a banner, "Loaded 2224 rows from all.tsv; 11 are known-missing... leaving 2213 available rows.", "Stratified sample: 15 rows/class -> 300 rows.", then 300 numbered lines like `[1/300] S04_E01_103 (Leave)`, then "DONE. Wrote 300 rows to data\mintrec_multimodal\index.csv". This involves a real network download (~330MB) plus 300 clips × (1 audio extract + 1 duration probe + 5 frame extracts) = 2,100 `ffmpeg`/`ffprobe` subprocess calls, so it will take several minutes — that's expected and already covered by the size-based confirmation above, not a violation of the "confirm before >1 minute" training-time rule (this is data acquisition, not training).

- [ ] **Step 3: Verify the generated files**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import wave
from pathlib import Path
import pandas as pd

root = Path('.')
idx = pd.read_csv(root / 'data' / 'mintrec_multimodal' / 'index.csv')
assert len(idx) == 300, f'expected 300 rows, got {len(idx)}'
assert list(idx.columns) == ['sample_id', 'text', 'intent', 'season', 'episode', 'clip', 'audio_path', 'frame_dir']
assert idx['intent'].nunique() == 20, f'expected 20 classes, got {idx[\"intent\"].nunique()}'
assert (idx['intent'].value_counts() == 15).all(), 'expected exactly 15 rows per class'

sample = idx.iloc[0]
audio_path = root / sample['audio_path']
with wave.open(str(audio_path), 'rb') as w:
    assert w.getnchannels() == 1, f'expected mono, got {w.getnchannels()} channels'
    assert w.getframerate() == 16000, f'expected 16kHz, got {w.getframerate()}'
    assert w.getnframes() > 0

frame_dir = root / sample['frame_dir']
frames = sorted(frame_dir.glob('frame_*.jpg'))
assert len(frames) == 5, f'expected 5 frames, got {len(frames)}'
for f in frames:
    assert f.stat().st_size > 0

print('All checks passed.')
print(idx.head(3).to_string(index=False))
"
```

Expected: `All checks passed.` followed by a 3-row preview.

- [ ] **Step 4: Verify re-running is idempotent (skips already-downloaded clips)**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
from pathlib import Path
p = Path('data/mintrec_multimodal/raw_clips')
mtimes_before = {f: f.stat().st_mtime for f in p.rglob('*.mp4')}
print('tracked', len(mtimes_before), 'clip mtimes')
"
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/generate_mintrec_multimodal.py
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
from pathlib import Path
p = Path('data/mintrec_multimodal/raw_clips')
mtimes_after = {f: f.stat().st_mtime for f in p.rglob('*.mp4')}
print('tracked', len(mtimes_after), 'clip mtimes')
"
```

Expected: the second run finishes in seconds (no new downloads — every `[i/300]` line still prints since the loop always runs, but `download_clip` returns immediately for existing files), and the clip file count/mtimes are unchanged between the two checks.

- [ ] **Step 5: Commit**

Use the `commit` skill. Stage `src/generate_mintrec_multimodal.py` explicitly — not `data/mintrec_multimodal/` (gitignored). Suggested message subject: `feat: add real MIntRec data acquisition for Milestone 2`.

---

## Task 2: Extract frozen encoder embeddings

> **Before dispatching this task: confirm with the human that it's OK to (a) install `torchvision` (small package) and (b) download `wav2vec2-base` (~360MB) and `ResNet18` (~45MB) pretrained weights (~405MB combined).** Do not proceed without that confirmation.

**Files:**
- Create: `src/extract_mintrec_embeddings.py`

**Interfaces:**
- Consumes: `data/mintrec_multimodal/index.csv` and the `audio_path`/`frame_dir` files it points to (from Task 1). Consumes the WAV format guarantee from Task 1 (mono, 16kHz, 16-bit PCM).
- Produces: `data/mintrec_multimodal/embeddings/{sample_id}.npz`, each with an `audio` key (float array, shape `(768,)`) and a `video` key (float array, shape `(512,)`).

- [ ] **Step 1: Install `torchvision` (after confirming with the human)**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi pip install torchvision
```

Expected: installs successfully (should resolve a version compatible with the already-installed `torch 2.13.0`).

- [ ] **Step 2: Write `src/extract_mintrec_embeddings.py`**

```python
"""
Milestone 2, increment 2: run frozen pretrained encoders over the real
MIntRec audio/frames and cache the resulting embeddings.
=================================================================
GOAL OF THIS FILE
    src/generate_mintrec_multimodal.py downloaded real clips and extracted
    raw audio (WAV) and video frames (JPEG) from them. This file is where
    those raw files actually meet a real pretrained model:

        audio WAV     -> wav2vec2-base (frozen) -> mean-pool over time -> 768 numbers
        5 JPEG frames -> ResNet18 (frozen)       -> average over frames -> 512 numbers

    Both models are FROZEN: we only ever run them forward to get an
    embedding, never train/fine-tune them (matching CLAUDE.md's "prefer
    frozen encoders over fine-tuning" rule).

    Running two neural networks over hundreds of clips is real CPU work --
    running this ONCE and caching the result means src/train_mintrec_multimodal.py
    (which you'll likely re-run many times while experimenting) can stay
    fast, only ever reading these cached .npz files instead of re-running
    the encoders every time.

HOW TO RUN
    conda activate mmi
    python src/generate_mintrec_multimodal.py   # once, if you haven't already
    python src/extract_mintrec_embeddings.py

    First run downloads wav2vec2-base (~360MB) and ResNet18 (~45MB) weights
    -- confirm with Claude Code before running this the first time. After
    that first download, weights are cached by Hugging Face/torch and
    reused on future runs. Safe to re-run: samples that already have a
    cached embedding are skipped.

TERMS YOU'LL SEE
    - embedding        : a fixed-length list of numbers a neural network
                          produces to summarize its input -- the audio
                          encoder's embedding summarizes a whole clip in
                          768 numbers, the video encoder's summarizes one
                          frame in 512 numbers.
    - mean-pooling      : averaging a sequence of vectors (e.g. wav2vec2's
                          per-timestep outputs) down to a single vector.
    - penultimate layer : the layer just before a network's final
                          classification layer -- its output is a general-
                          purpose feature vector, not tied to ImageNet's
                          specific 1000 classes, which is why it's useful
                          as an embedding for a totally different task.
"""

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "mintrec_multimodal" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal" / "embeddings"

AUDIO_CKPT = "facebook/wav2vec2-base"  # self-supervised base model, no task-specific
                                        # head -- a general-purpose frozen audio
                                        # embedding, not tied to e.g. speech recognition.


def load_audio_encoder():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    model.eval()  # frozen: we never call .train() or run a backward pass
    return feature_extractor, model


def load_video_encoder():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # drop the final 1000-class layer; what's left
                                     # outputs ResNet18's 512-dim penultimate embedding
    model.eval()
    return model, weights.transforms()


def read_wav_as_array(wav_path):
    """Read a mono 16-bit PCM WAV file into a float32 numpy array in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def embed_audio(wav_path, feature_extractor, model):
    audio = read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (1, time_steps, 768) -- mean-pool over time
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.numpy()


def embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.mean(embeddings, axis=0)


def main():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the real MIntRec multimodal data first:\n"
            "    python src/generate_mintrec_multimodal.py\n"
        )

    print("=" * 70)
    print("EXTRACT REAL MINTREC EMBEDDINGS (frozen wav2vec2-base + ResNet18)")
    print("=" * 70)

    df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(df)} rows from {INDEX_PATH.name}")

    print("\nLoading frozen audio encoder (wav2vec2-base)...")
    feature_extractor, audio_model = load_audio_encoder()
    print("Loading frozen video encoder (ResNet18)...")
    video_model, video_transforms = load_video_encoder()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        out_path = EMBEDDINGS_DIR / f"{row.sample_id}.npz"
        if out_path.exists():
            continue
        print(f"[{i}/{len(df)}] {row.sample_id}")
        audio_embedding = embed_audio(PROJECT_ROOT / row.audio_path, feature_extractor, audio_model)
        video_embedding = embed_video(PROJECT_ROOT / row.frame_dir, video_model, video_transforms)
        np.savez(out_path, audio=audio_embedding, video=video_embedding)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote embeddings for {len(df)} samples to")
    print(f"{EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}")
    print("Next: python src/train_mintrec_multimodal.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

If `from PIL import Image` fails (Pillow should already be present as a `matplotlib` dependency, but if not): report `NEEDS_CONTEXT` rather than silently installing it — this is a small, safe install, but per this project's rules, confirm before installing anything, even small packages.

- [ ] **Step 3: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/extract_mintrec_embeddings.py
```

Expected: a banner, the loaded-row count, "Loading frozen audio encoder..." / "Loading frozen video encoder..." (first run downloads the weights here), then 300 numbered lines, then "DONE. Wrote embeddings for 300 samples to data\mintrec_multimodal\embeddings". CPU inference over 300 clips (each: 1 wav2vec2 forward pass + 5 ResNet18 forward passes) will take a few minutes — expected, not a violation of the training-time rule (this is one-time embedding extraction, not a training loop).

- [ ] **Step 4: Verify the generated embeddings**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import numpy as np
import pandas as pd
from pathlib import Path

root = Path('.')
idx = pd.read_csv(root / 'data' / 'mintrec_multimodal' / 'index.csv')
emb_dir = root / 'data' / 'mintrec_multimodal' / 'embeddings'
emb_files = list(emb_dir.glob('*.npz'))
assert len(emb_files) == len(idx), f'expected {len(idx)} embeddings, got {len(emb_files)}'

sample = np.load(emb_dir / f'{idx.iloc[0][\"sample_id\"]}.npz')
assert sample['audio'].shape == (768,), f'unexpected audio embedding shape {sample[\"audio\"].shape}'
assert sample['video'].shape == (512,), f'unexpected video embedding shape {sample[\"video\"].shape}'
assert not np.isnan(sample['audio']).any()
assert not np.isnan(sample['video']).any()

print('All checks passed.')
"
```

Expected: `All checks passed.`

- [ ] **Step 5: Commit**

Use the `commit` skill. Stage `src/extract_mintrec_embeddings.py` explicitly — not `data/mintrec_multimodal/embeddings/` (gitignored). Suggested message subject: `feat: add frozen encoder embedding extraction for Milestone 2`.

---

## Task 3: Fusion training script on real embeddings

**Files:**
- Create: `src/train_mintrec_multimodal.py`

**Interfaces:**
- Consumes: `data/mintrec_multimodal/index.csv` (from Task 1) and `data/mintrec_multimodal/embeddings/{sample_id}.npz` with `audio` (768,) / `video` (512,) keys (from Task 2).
- Produces: `results/mintrec_multimodal_predictions.csv` (columns `text, true, predicted`) and `results/mintrec_multimodal_confusion_matrix.csv` (20x20, labeled rows/columns — following `src/train_mintrec.py`'s precedent of CSV export instead of an ASCII table, since 20 classes is too wide for a terminal, unlike the toy pipeline's 8 classes).

- [ ] **Step 1: Write `src/train_mintrec_multimodal.py`**

```python
"""
Milestone 2, increment 2: fuse text + REAL audio + REAL video embeddings.
=================================================================
GOAL OF THIS FILE
    Same load -> split -> fuse -> train -> evaluate loop as
    src/train_multimodal_toy.py, but the audio/video features are no
    longer hand-written stand-ins (FFT stats, color stats) -- they're
    real embeddings from frozen pretrained models (wav2vec2-base,
    ResNet18), precomputed by src/extract_mintrec_embeddings.py.

    Unlike the toy pipeline's engineered 96-row, 8-class, deliberately
    learnable data, this is real dialogue across 20 real intent classes.
    The accuracy here is genuinely unknown ahead of time -- it may or may
    not beat Milestone 0's 0.550 text-only baseline, and that's an honest
    result either way, not something to chase.

HOW TO RUN
    conda activate mmi
    python src/generate_mintrec_multimodal.py     # once
    python src/extract_mintrec_embeddings.py       # once
    python src/train_mintrec_multimodal.py

    Finishes in a couple of seconds -- all the expensive work (downloading,
    running the encoders) already happened in the two scripts above; this
    one only loads their cached output.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "mintrec_multimodal" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data():
    """Read the real MIntRec multimodal index and show what we're working with."""
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the real MIntRec multimodal data first:\n"
            "    python src/generate_mintrec_multimodal.py\n"
        )
    if not EMBEDDINGS_DIR.exists() or not any(EMBEDDINGS_DIR.glob("*.npz")):
        raise SystemExit(
            f"\nNo embeddings found in {EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}.\n"
            "Extract embeddings first:\n"
            "    python src/extract_mintrec_embeddings.py\n"
        )
    print("=" * 70)
    print("STEP 1  LOAD THE DATA")
    print("=" * 70)
    df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(df)} samples from {INDEX_PATH.name}")
    print("\nHow many examples we have per intent:")
    print(df["intent"].value_counts().to_string())
    return df


def split_data(df):
    """Same stratified train/val/test split as every prior script."""
    print("\n" + "=" * 70)
    print("STEP 2  SPLIT INTO TRAIN / VALIDATION / TEST")
    print("=" * 70)
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["intent"]
    )
    train, val = train_test_split(
        train_val, test_size=0.1875, random_state=42, stratify=train_val["intent"]
    )
    print(f"train: {len(train)} samples  (model learns from these)")
    print(f"val:   {len(val)} samples  (reserved for tuning)")
    print(f"test:  {len(test)} samples  (final honest judgement)")
    return train, val, test


def build_audio_features(df):
    """Load each sample's precomputed 768-dim wav2vec2 embedding."""
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["audio"] for sid in df["sample_id"]])


def build_video_features(df):
    """Load each sample's precomputed 512-dim ResNet18 embedding."""
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["video"] for sid in df["sample_id"]])


def build_features(df, tfidf, fit):
    """Text (TF-IDF) + audio + video raw (unscaled) feature matrices for one split."""
    if fit:
        text_feats = tfidf.fit_transform(df["text"]).toarray()
    else:
        text_feats = tfidf.transform(df["text"]).toarray()
    audio_feats = build_audio_features(df)
    video_feats = build_video_features(df)
    return text_feats, audio_feats, video_feats


def train_fused(train):
    """Build the fused feature vectors and train Logistic Regression on them."""
    print("\n" + "=" * 70)
    print("STEP 3  EXTRACT FEATURES FROM ALL THREE MODALITIES AND FUSE")
    print("=" * 70)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    text_feats, audio_feats, video_feats = build_features(train, tfidf, fit=True)
    fused = np.hstack([text_feats, audio_feats, video_feats])
    print(
        f"Fused feature vector size: {fused.shape[1]} "
        f"(text={text_feats.shape[1]}, audio=768, video=512)"
    )

    scaler = StandardScaler()
    fused_scaled = scaler.fit_transform(fused)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(fused_scaled, train["intent"])
    print("Training done.")
    return tfidf, scaler, clf


def evaluate(tfidf, scaler, clf, test):
    """Judge the fused model on the held-out test set."""
    print("\n" + "=" * 70)
    print("STEP 4  EVALUATE ON THE TEST SET")
    print("=" * 70)

    text_feats, audio_feats, video_feats = build_features(test, tfidf, fit=False)
    fused = np.hstack([text_feats, audio_feats, video_feats])
    fused_scaled = scaler.transform(fused)
    preds = clf.predict(fused_scaled)
    truth = test["intent"].tolist()

    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro")
    print(f"Accuracy : {acc:.3f}   (share of predictions that were exactly right)")
    print(f"Macro-F1 : {macro_f1:.3f}   (fairer average across all intents)")

    print("\nPer-class report (precision/recall/F1 for each intent):")
    print(classification_report(truth, preds, zero_division=0))

    RESULTS_DIR.mkdir(exist_ok=True)

    labels = sorted(test["intent"].unique())
    cm = confusion_matrix(truth, preds, labels=labels)
    cm_path = RESULTS_DIR / "mintrec_multimodal_confusion_matrix.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path)
    print(f"Confusion matrix (20x20, too wide for terminal) saved to {cm_path.relative_to(PROJECT_ROOT)}")

    out = RESULTS_DIR / "mintrec_multimodal_predictions.csv"
    pd.DataFrame({"text": test["text"], "true": truth, "predicted": preds}).to_csv(out, index=False)
    print(f"Saved every test prediction to {out.relative_to(PROJECT_ROOT)}")

    return acc, macro_f1


def evaluate_config(name, train_feats, test_feats, train_labels, test_labels, use_scaler):
    """One row of the ablation table: fit+evaluate a single feature matrix, with or without scaling."""
    if use_scaler:
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    print(f"{name:35s} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    return acc, f1


def run_ablation(train, test, tfidf):
    """
    Does scaling actually help on REAL data? Does fusion actually help over
    a single modality? The toy pipeline found scaling hurt there -- that
    was a toy-data artifact (one modality was artificially too clean), not
    a general finding. Check again here rather than assuming either way.
    """
    print("\n" + "=" * 70)
    print("ABLATION: does scaling help? does fusion help over single modalities?")
    print("=" * 70)

    text_train, audio_train, video_train = build_features(train, tfidf, fit=False)
    text_test, audio_test, video_test = build_features(test, tfidf, fit=False)
    train_labels, test_labels = train["intent"], test["intent"]

    fused_train = np.hstack([text_train, audio_train, video_train])
    fused_test = np.hstack([text_test, audio_test, video_test])

    evaluate_config("text-only, no scaler", text_train, text_test, train_labels, test_labels, False)
    evaluate_config("text-only, with scaler", text_train, text_test, train_labels, test_labels, True)
    evaluate_config("audio-only, with scaler", audio_train, audio_test, train_labels, test_labels, True)
    evaluate_config("video-only, with scaler", video_train, video_test, train_labels, test_labels, True)
    evaluate_config("fused, no scaler", fused_train, fused_test, train_labels, test_labels, False)
    evaluate_config("fused, with scaler (ships)", fused_train, fused_test, train_labels, test_labels, True)


def main():
    df = load_data()
    train, val, test = split_data(df)
    tfidf, scaler, clf = train_fused(train)
    evaluate(tfidf, scaler, clf, test)
    run_ablation(train, test, tfidf)

    print("\n" + "=" * 70)
    print("DONE. Compare this accuracy to Milestone 0's text-only 0.550 -- ask Claude Code:")
    print("  - 'Did fusing real audio/video actually help here, or hurt?'")
    print("  - 'Which intents does the real fused model get right that text-only got wrong?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/train_mintrec_multimodal.py
```

Expected: STEP 1-4 banners; a "Fused feature vector size: N (text=X, audio=768, video=512)" line; `Accuracy`/`Macro-F1` lines (the real, previously-unknown result); a classification report; confirmation the confusion matrix and predictions were saved; then the 6-row ablation table; then the closing banner. Finishes in a few seconds (all expensive work already happened in Tasks 1-2).

- [ ] **Step 3: Verify the results are real and self-consistent**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd
from sklearn.metrics import accuracy_score

preds = pd.read_csv('results/mintrec_multimodal_predictions.csv')
assert len(preds) == 60, f'expected 60 test rows (20% of 300), got {len(preds)}'
assert list(preds.columns) == ['text', 'true', 'predicted']
acc = accuracy_score(preds['true'], preds['predicted'])
print(f'accuracy from saved predictions: {acc:.3f}')

cm = pd.read_csv('results/mintrec_multimodal_confusion_matrix.csv', index_col=0)
assert cm.shape == (20, 20), f'expected a 20x20 confusion matrix, got {cm.shape}'
assert cm.values.sum() == 60, f'confusion matrix should sum to the 60 test predictions, got {cm.values.sum()}'
print('OK: results files are internally consistent.')
"
```

Expected: `OK: results files are internally consistent.` (No accuracy floor assertion here, unlike Task 2 of the toy-fusion plan — per this task's own docstring and this plan's Global Constraints, the real accuracy is genuinely unknown and must be reported as-is, not gated on beating 0.550.)

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/train_mintrec_multimodal.py` explicitly — not `results/` (gitignored). Suggested message subject: `feat: add real MIntRec fusion training script for Milestone 2`.

---

## Task 4: Tutorial notebook

**Files:**
- Create: `notebooks/milestone2_mintrec_multimodal.ipynb`
- Create (scratchpad only, not committed): a Python builder script filling the notebook's cells, same throwaway pattern used for the three prior notebooks.

**Interfaces:**
- Consumes: `data/mintrec_multimodal/index.csv` (Task 1) and `data/mintrec_multimodal/embeddings/*.npz` (Task 2) — loads cached embeddings directly, does NOT reload `wav2vec2`/`ResNet18` itself, so the notebook stays fast (seconds) like every prior notebook, matching Task 3's `train_mintrec_multimodal.py` logic exactly.

- [ ] **Step 1: Scaffold the notebook from the tutorial template**

```bash
python "C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py" --kind tutorial --title "Milestone 2 - Real MIntRec Multimodal Fusion" --out "notebooks/milestone2_mintrec_multimodal.ipynb"
```

Expected: `Wrote ...notebooks\milestone2_mintrec_multimodal.ipynb using kind=tutorial.`

- [ ] **Step 2: Write the cell-filling builder script**

Save to the scratchpad (e.g. `build_m2_mintrec_notebook.py`), adjusting `NB_PATH` if your scratchpad differs:

```python
"""Builds notebooks/milestone2_mintrec_multimodal.ipynb by filling the scaffolded template."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\dbest\Downloads\multimodal-intent-inference\multimodal-intent-inference\notebooks\milestone2_mintrec_multimodal.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = [
    md(
        "# Tutorial: Milestone 2 - Real MIntRec Multimodal Fusion",
        "",
        "Audience:",
        "- You've completed the toy fusion notebook (`milestone2_toy_fusion.ipynb`) and understand concatenation",
        "  fusion, StandardScaler, and why an ablation matters. This notebook runs the same idea on REAL data.",
        "",
        "Prerequisites:",
        "- Run these once first, in order (each is a real download/compute step, not instant):",
        "  `python src/generate_mintrec_multimodal.py`, then `python src/extract_mintrec_embeddings.py`.",
        "",
        "Learning goals:",
        "- By the end, you can explain what a frozen pretrained encoder embedding is, and read a real",
        "  (not engineered) result comparing text-only vs. fused accuracy on real dialogue.",
        "",
        "**What changed from the toy notebook:** real short video clips (real dialogue from the Superstore TV",
        "show, from the MIntRec dataset) instead of invented sine waves and solid colors; real frozen",
        "`wav2vec2-base` (audio) and `ResNet18` (video) embeddings instead of hand-written FFT/color-stat",
        "features; 20 real, imbalanced intent classes instead of 8 tidy invented ones. The concatenation +",
        "scaling + Logistic Regression structure is otherwise the same.",
    ),
    md(
        "## Outline",
        "",
        "1. Setup",
        "2. Step 1 - Load the real MIntRec multimodal index",
        "3. Step 2 - Split into train / validation / test",
        "4. Step 3 - Load precomputed real embeddings and fuse them",
        "5. Step 4 - Evaluate, and compare against Milestone 0",
        "6. Ablation - does scaling help on real data? does fusion help?",
        "7. Look at the mistakes",
        "8. Exercises",
        "9. Pitfalls and extensions",
    ),
    code(
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix",
        "import matplotlib.pyplot as plt",
        "",
        "NOTEBOOK_DIR = Path.cwd()",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR",
        "INDEX_PATH = PROJECT_ROOT / \"data\" / \"mintrec_multimodal\" / \"index.csv\"",
        "EMBEDDINGS_DIR = PROJECT_ROOT / \"data\" / \"mintrec_multimodal\" / \"embeddings\"",
        "RESULTS_DIR = PROJECT_ROOT / \"results\"",
        "INDEX_PATH",
    ),
    md(
        "## Step 1 - Load the real MIntRec multimodal index",
        "",
        "Unlike the toy version, this data is real dialogue: 20 real intent classes, genuinely imbalanced,",
        "with real `season`/`episode`/`clip` provenance for every row.",
    ),
    code(
        "df = pd.read_csv(INDEX_PATH)",
        "print(f\"Loaded {len(df)} samples from {INDEX_PATH.name}\")",
        "print(f\"Number of distinct intent classes: {df['intent'].nunique()}\")",
        "df[\"intent\"].value_counts()",
    ),
    code(
        "# A real sample: its text, its actual clip location, and its extracted frames.",
        "sample = df.iloc[0]",
        "print(f\"Sample: {sample['text']!r} -> intent={sample['intent']}\")",
        "print(f\"From: {sample['season']} {sample['episode']} clip {sample['clip']}\")",
        "",
        "frame_paths = sorted((PROJECT_ROOT / sample[\"frame_dir\"]).glob(\"frame_*.jpg\"))",
        "fig, axes = plt.subplots(1, len(frame_paths), figsize=(12, 3))",
        "for ax, fp in zip(axes, frame_paths):",
        "    ax.imshow(plt.imread(fp))",
        "    ax.axis(\"off\")",
        "fig.suptitle(\"5 evenly-spaced frames extracted from this clip\")",
        "plt.show()",
    ),
    md(
        "## Step 2 - Split into train / validation / test",
        "",
        "Identical split logic to every prior notebook.",
    ),
    code(
        "train_val, test = train_test_split(",
        "    df, test_size=0.20, random_state=42, stratify=df[\"intent\"]",
        ")",
        "train, val = train_test_split(",
        "    train_val, test_size=0.1875, random_state=42, stratify=train_val[\"intent\"]",
        ")",
        "",
        "print(f\"train: {len(train)} samples\")",
        "print(f\"val:   {len(val)} samples\")",
        "print(f\"test:  {len(test)} samples\")",
    ),
    md(
        "## Step 3 - Load precomputed real embeddings and fuse them",
        "",
        "`src/extract_mintrec_embeddings.py` already ran `wav2vec2-base` and `ResNet18` over every clip and",
        "cached the results. We just load those cached numbers here -- no neural network runs in this notebook,",
        "which is why it stays fast even though real encoders were involved upstream.",
    ),
    code(
        "def build_audio_features(frame_df):",
        "    return np.array([np.load(EMBEDDINGS_DIR / f\"{sid}.npz\")[\"audio\"] for sid in frame_df[\"sample_id\"]])",
        "",
        "",
        "def build_video_features(frame_df):",
        "    return np.array([np.load(EMBEDDINGS_DIR / f\"{sid}.npz\")[\"video\"] for sid in frame_df[\"sample_id\"]])",
        "",
        "",
        "def build_features(frame_df, tfidf, fit):",
        "    text_feats = tfidf.fit_transform(frame_df[\"text\"]).toarray() if fit else tfidf.transform(frame_df[\"text\"]).toarray()",
        "    return text_feats, build_audio_features(frame_df), build_video_features(frame_df)",
    ),
    code(
        "tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)",
        "text_train, audio_train, video_train = build_features(train, tfidf, fit=True)",
        "fused_train_raw = np.hstack([text_train, audio_train, video_train])",
        "print(f\"Fused feature vector size: {fused_train_raw.shape[1]} \"",
        "      f\"(text={text_train.shape[1]}, audio=768, video=512)\")",
        "",
        "scaler = StandardScaler()",
        "fused_train = scaler.fit_transform(fused_train_raw)",
        "",
        "clf = LogisticRegression(max_iter=1000)",
        "clf.fit(fused_train, train[\"intent\"])",
        "print(\"Training done.\")",
    ),
    md(
        "## Step 4 - Evaluate, and compare against Milestone 0",
        "",
        "This accuracy is genuinely unknown ahead of time -- unlike the toy notebook's engineered 0.700, there's",
        "no guarantee real fusion beats Milestone 0's text-only baseline. Either result is a real, honest finding.",
    ),
    code(
        "text_test, audio_test, video_test = build_features(test, tfidf, fit=False)",
        "fused_test = scaler.transform(np.hstack([text_test, audio_test, video_test]))",
        "preds = clf.predict(fused_test)",
        "truth = test[\"intent\"].tolist()",
        "",
        "acc = accuracy_score(truth, preds)",
        "macro_f1 = f1_score(truth, preds, average=\"macro\")",
        "print(f\"Accuracy : {acc:.3f}\")",
        "print(f\"Macro-F1 : {macro_f1:.3f}\")",
    ),
    code(
        "print(classification_report(truth, preds, zero_division=0))",
    ),
    code(
        "labels = sorted(test[\"intent\"].unique())",
        "cm = confusion_matrix(truth, preds, labels=labels)",
        "",
        "fig, ax = plt.subplots(figsize=(10, 9))",
        "im = ax.imshow(cm, cmap=\"Blues\")",
        "ax.set_xticks(range(len(labels)), labels, rotation=90)",
        "ax.set_yticks(range(len(labels)), labels)",
        "ax.set_xlabel(\"predicted\")",
        "ax.set_ylabel(\"true\")",
        "ax.set_title(\"Real MIntRec fusion confusion matrix (20 classes)\")",
        "fig.colorbar(im, ax=ax, shrink=0.8, label=\"count\")",
        "fig.tight_layout()",
        "plt.show()",
    ),
    md(
        "### Compare against Milestone 0 and the toy fusion notebook",
    ),
    code(
        "comparison = pd.DataFrame([",
        "    {\"model\": \"Text-only, toy 8-class (M0)\", \"accuracy\": 0.550, \"macro_f1\": 0.521},",
        "    {\"model\": \"Text-only, real 20-class (M1)\", \"accuracy\": 0.508, \"macro_f1\": 0.409},",
        "    {\"model\": \"Fused, toy 8-class, synthetic (M2 toy)\", \"accuracy\": 0.700, \"macro_f1\": 0.683},",
        "    {\"model\": \"Fused, real 20-class, real encoders (M2 real)\", \"accuracy\": acc, \"macro_f1\": macro_f1},",
        "]).set_index(\"model\")",
        "comparison",
    ),
    md(
        "## Ablation - does scaling help on real data? does fusion help?",
        "",
        "The toy notebook found `StandardScaler` measurably *hurt* accuracy there -- but that was because one",
        "toy modality (video) was artificially made almost perfectly separable. Real audio/video embeddings",
        "won't have that property, so this needs checking again here rather than assumed either way.",
    ),
    code(
        "def evaluate_config(name, train_feats, test_feats, train_labels, test_labels, use_scaler):",
        "    if use_scaler:",
        "        s = StandardScaler()",
        "        train_feats = s.fit_transform(train_feats)",
        "        test_feats = s.transform(test_feats)",
        "    c = LogisticRegression(max_iter=1000)",
        "    c.fit(train_feats, train_labels)",
        "    p = c.predict(test_feats)",
        "    a = accuracy_score(test_labels, p)",
        "    f = f1_score(test_labels, p, average=\"macro\")",
        "    print(f\"{name:35s} accuracy={a:.3f}  macro-F1={f:.3f}\")",
        "    return a, f",
        "",
        "",
        "fused_test_raw = np.hstack([text_test, audio_test, video_test])",
        "train_labels, test_labels = train[\"intent\"], test[\"intent\"]",
        "",
        "evaluate_config(\"text-only, no scaler\", text_train, text_test, train_labels, test_labels, False)",
        "evaluate_config(\"text-only, with scaler\", text_train, text_test, train_labels, test_labels, True)",
        "evaluate_config(\"audio-only, with scaler\", audio_train, audio_test, train_labels, test_labels, True)",
        "evaluate_config(\"video-only, with scaler\", video_train, video_test, train_labels, test_labels, True)",
        "evaluate_config(\"fused, no scaler\", fused_train_raw, fused_test_raw, train_labels, test_labels, False)",
        "evaluate_config(\"fused, with scaler (ships)\", fused_train_raw, fused_test_raw, train_labels, test_labels, True)",
    ),
    md(
        "## Look at the mistakes",
    ),
    code(
        "results_df = pd.DataFrame({\"text\": test[\"text\"], \"true\": truth, \"predicted\": preds})",
        "",
        "RESULTS_DIR.mkdir(exist_ok=True)",
        "out_path = RESULTS_DIR / \"mintrec_multimodal_predictions.csv\"",
        "results_df.to_csv(out_path, index=False)",
        "print(f\"Saved every test prediction to {out_path.relative_to(PROJECT_ROOT)}\")",
        "",
        "mistakes = results_df[results_df[\"true\"] != results_df[\"predicted\"]]",
        "print(f\"\\n{len(mistakes)} of {len(results_df)} test sentences were misclassified \"",
        "      f\"({len(mistakes)/len(results_df):.1%}).\")",
        "mistakes.head(10)",
    ),
    md(
        "## Exercises",
        "",
        "1. Compare this notebook's confusion matrix to Milestone 1's text-only 20-class confusion matrix",
        "   (`notebooks/milestone1_mintrec_text.ipynb`). Are the same intent pairs confused (e.g. `Complain` vs.",
        "   `Criticize`), or did adding audio/video change which mistakes the model makes?",
        "2. `src/generate_mintrec_multimodal.py --full` would use all 2,213 available clips instead of this",
        "   notebook's 300-clip subset. Without running it (it's a ~2.3GB download, needing its own",
        "   confirmation), predict: would you expect accuracy to go up, down, or be unpredictable with ~7x more",
        "   training data per class? Why?",
    ),
    md(
        "## Pitfalls and extensions",
        "",
        "**Common mistake:** assuming a result from synthetic/toy data transfers to real data. The toy",
        "notebook's \"scaling hurts\" finding didn't necessarily hold here -- see the ablation above for what",
        "actually happened on real embeddings. Always re-check assumptions when the underlying data changes.",
        "",
        "**Extension:** this pipeline uses a 300-clip stratified subset for speed. Once you're ready to commit",
        "to a larger download and longer processing time, `--full` on both",
        "`src/generate_mintrec_multimodal.py` and a rerun of `src/extract_mintrec_embeddings.py` would use all",
        "2,213 available real clips -- confirm the ~2.3GB download and the longer embedding-extraction run with",
        "Claude Code first, per this project's rules.",
        "",
        "**Extension:** both encoders are frozen here (no fine-tuning), per `CLAUDE.md`'s preference while",
        "prototyping. A natural next step once this pipeline is trusted: fine-tune just the classifier head",
        "further, or unfreeze the last layer or two of one encoder, and compare.",
    ),
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells to {NB_PATH}")
```

- [ ] **Step 3: Run the builder script**

```bash
python <path-to-scratchpad>/build_m2_mintrec_notebook.py
```

Expected: `Wrote 20 cells to ...notebooks\milestone2_mintrec_multimodal.ipynb`.

- [ ] **Step 4: Execute the notebook top-to-bottom**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi jupyter nbconvert --to notebook --execute --inplace "notebooks/milestone2_mintrec_multimodal.ipynb"
```

Expected: `[NbConvertApp] Writing ... bytes to ...notebooks\milestone2_mintrec_multimodal.ipynb` with no Python traceback (the usual symlink/TCP-kernel warnings are expected and harmless). Should finish in well under a minute — this notebook only loads cached embeddings, it doesn't run either neural encoder.

- [ ] **Step 5: Verify no errors in the executed notebook**

```bash
python -c "
import json
nb = json.load(open('notebooks/milestone2_mintrec_multimodal.ipynb', encoding='utf-8'))
errors = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                errors += 1
                print(i, 'ERROR:', o.get('ename'), o.get('evalue')[:200])
print('total errors:', errors)
assert errors == 0
"
```

Expected: `total errors: 0`.

- [ ] **Step 6: Commit**

Use the `commit` skill. Stage `notebooks/milestone2_mintrec_multimodal.ipynb` only (the builder script stays in the scratchpad). Suggested message subject: `feat: add real MIntRec multimodal fusion notebook`.

---

## Final check

After Task 4, read back the real accuracy/macro-F1 from Task 3's run and the notebook's comparison table. Update `CLAUDE.md`'s M2 entry to reflect that it's now fully done (real data + real frozen encoders + concatenation fusion), replacing the "toy sanity check only" note from the first increment — but confirm the exact wording with the user first, the same way M1's and M2-increment-1's checkboxes were handled.
