# Milestone 4 Cross-Modal Dialogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cross-modal transformer (audio/video attending into text) that uses dialogue context (up to 4 preceding utterances) to classify the current utterance's intent, on 10 complete real MIntRec episodes.

**Architecture:** Five new files. Two data-preparation scripts (acquire 10 complete episodes + build dialogue windows; extract sequence-level — not pooled — frozen embeddings). A model file (cross-modal attention + dialogue self-attention, with a built-in ablation flag). A training script (episode-level split to prevent context leakage, early stopping, honest comparison against M0/M2/M3 plus a same-capacity no-cross-attention ablation). A narrated notebook.

**Tech Stack:** Python 3.11, `mmi` conda env. `torch`, `torchvision`, `transformers`, `huggingface_hub` (all already installed from M2/M3). No new dependencies.

## Global Constraints

- New files only — do not modify any M0-M3 file.
- Follow established code patterns: module docstrings (goal/how-to-run/terms), `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, `"=" * 70` banners, `load_data()`/`split_data()`/`evaluate()` naming, `SystemExit` with a clear pointer to the missing prerequisite script.
- No pytest suite — verification is run-the-script/smoke-test, read the output, check generated files.
- **This milestone's split is by EPISODE, not by individual utterance** (7 train / 1 val / 2 test episodes, exact lists given in Task 4) — this is a deliberate departure from every prior milestone's `train_test_split`-based split, necessary to prevent a target utterance's dialogue context from crossing into a different split than the target itself. Do not "fix" this to look like M0-M3's split pattern.
- **Report the real result honestly** — do not adjust hyperparameters to chase a particular number relative to M0 (0.508 on the larger real dataset)/M2 (0.117)/M3 (0.333). Per the lesson from M3's final review, always report the no-cross-attention ablation alongside the full model, not just the full model in isolation — this project got burned once already by a milestone crediting the wrong mechanism for an accuracy change.
- Training must be fast (small hidden_dim=128, ~350-400 training rows) — well under a minute per run is expected; note as a deviation in the implementer's report if a run unexpectedly takes several minutes.
- Data acquisition (Task 1) is a real ~560MB download across 10 episodes and ~2,500-3,500 `ffmpeg`/`ffprobe` subprocess calls (512 clips × ~5-7 calls each) — will take several minutes, which is expected for acquisition (not training) and does not need a separate confirmation gate under this milestone's autonomous-execution authorization.
- The project's `commit` skill (secret scan, draft message, staged-files review) is used for every commit, not raw `git commit`.
- Working directly on `master`, no worktree (matches all prior plans).
- Known environment quirks (see project memory / prior milestones' notes): `conda run -n mmi python -c "<multiline>"` fails on this machine (`NotImplementedError`) — use a temp `.py` file for any multi-line verification snippet. Never invoke the `mmi` env's `python.exe` directly (bypassing `conda run`) — it crashes with `STATUS_STACK_BUFFER_OVERRUN`; always go through `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python ...`.

---

## Task 1: Acquire dialogue data (10 complete episodes)

**Files:**
- Create: `src/generate_dialogue_data.py`

**Interfaces:**
- Produces: `data/dialogue/index.csv` with columns `sample_id, target_text, target_intent, season, episode, clip, context_sample_ids` (semicolon-joined, oldest-first, empty string if no context available — e.g. an episode's first utterance).
- Produces: `data/dialogue/raw_clips/{season}/{episode}/{clip}.mp4`, `data/dialogue/audio/{sample_id}.wav` (mono 16kHz 16-bit PCM), `data/dialogue/frames/{sample_id}/frame_{0..4}.jpg`.
- `sample_id` is `{season}_{episode}_{clip}`, matching M2/M3's convention.

- [ ] **Step 1: Write `src/generate_dialogue_data.py`**

```python
"""
Milestone 4: acquire real MIntRec dialogue data (complete episodes) and
build context windows for dialogue-aware intent classification.
=================================================================
GOAL OF THIS FILE
    Milestones 2-3 used a STRATIFIED SAMPLE of clips (15 per intent class,
    drawn from across all 43 episodes) -- deliberately not contiguous
    dialogue. That's the wrong shape of data for THIS milestone: dialogue
    context needs complete, temporally-ordered runs of utterances within
    an episode, so a model can look at what was said just before the
    utterance it's trying to classify.

    This file downloads 10 COMPLETE episodes (~512 clips total) instead of
    a stratified sample, and builds a "dialogue window" per utterance: its
    own clip plus up to 4 preceding clips from the same episode, ordered
    by clip number (used here as a proxy for chronological order -- see
    docs/superpowers/specs/2026-08-06-milestone4-crossmodal-dialogue-design.md
    for why, and the caveat that this is an unverified assumption).

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py

    Downloads ~560MB across 10 episodes -- a real download, similar in
    scale to Milestone 2's data acquisition, and will take several
    minutes (512 clips x ~7 ffmpeg/ffprobe calls each). Safe to re-run:
    already-downloaded clips are skipped.

TERMS YOU'LL SEE
    - dialogue window : the current utterance plus a handful of the
                          utterances immediately before it in the same
                          conversation
    - context          : the "before" utterances in a dialogue window --
                          information the model gets to see but isn't
                          being asked to classify
    - target            : the one utterance in a window whose intent the
                          model is actually being asked to predict
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_TSV_PATH = PROJECT_ROOT / "data" / "mintrec" / "all.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dialogue"

REPO_ID = "THU-IAR/MIntRec"
AUDIO_SAMPLE_RATE = 16000
N_FRAMES = 5
CONTEXT_SIZE = 4

# 10 complete episodes selected for a mix of season variety and per-episode
# class diversity (13-18 of the 20 intents present per episode) -- see the
# design spec for the full table and reasoning.
SELECTED_EPISODES = [
    ("S04", "E16"), ("S04", "E04"), ("S04", "E01"),
    ("S05", "E19"), ("S05", "E18"), ("S05", "E20"),
    ("S06", "E01"), ("S06", "E03"), ("S06", "E04"), ("S06", "E02"),
]

# Same known-missing rows as Milestone 2 (verified against the HF mirror's
# file listing) -- none of these fall inside SELECTED_EPISODES, but the
# check is kept for safety/consistency in case episodes are ever changed.
KNOWN_MISSING = {
    ("S05", "E07", 96), ("S05", "E15", 83), ("S05", "E15", 82),
    ("S05", "E07", 82), ("S05", "E15", 86), ("S05", "E09", 91),
    ("S05", "E07", 87), ("S05", "E15", 85), ("S05", "E15", 94),
    ("S05", "E15", 8), ("S05", "E09", 90),
}


def load_episode_rows():
    """Read all.tsv, keep only rows from SELECTED_EPISODES, sorted by
    (season, episode, clip) so clip order within an episode is ascending."""
    df = pd.read_csv(ALL_TSV_PATH, sep="\t")
    is_missing = df.apply(lambda r: (r["season"], r["episode"], r["clip"]) in KNOWN_MISSING, axis=1)
    df = df[~is_missing]
    mask = df.apply(lambda r: (r["season"], r["episode"]) in SELECTED_EPISODES, axis=1)
    selected = df[mask].sort_values(["season", "episode", "clip"]).reset_index(drop=True)
    print(f"Loaded {len(selected)} rows across {len(SELECTED_EPISODES)} episodes.")
    return selected


def build_windows(episode_rows):
    """For each row, gather up to CONTEXT_SIZE preceding sample_ids from
    the SAME episode (already sorted by clip ascending), oldest-first."""
    rows = []
    for (season, episode), group in episode_rows.groupby(["season", "episode"], sort=False):
        group = group.reset_index(drop=True)
        sample_ids = [f"{season}_{episode}_{row.clip}" for row in group.itertuples(index=False)]
        for i, row in enumerate(group.itertuples(index=False)):
            context_ids = sample_ids[max(0, i - CONTEXT_SIZE):i]  # oldest-first, up to 4
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "target_text": row.text,
                    "target_intent": row.label,
                    "season": season,
                    "episode": episode,
                    "clip": row.clip,
                    "context_sample_ids": ";".join(context_ids),
                }
            )
    return pd.DataFrame(rows)


def download_clip(season, episode, clip):
    """Download one raw clip from the Hugging Face mirror, skipping if already local."""
    local_path = OUTPUT_DIR / "raw_clips" / season / episode / f"{clip}.mp4"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
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
    print("=" * 70)
    print("GENERATE DIALOGUE DATA (10 complete episodes)")
    print("=" * 70)

    episode_rows = load_episode_rows()
    windows = build_windows(episode_rows)

    audio_paths, frame_dirs = [], []
    for i, row in enumerate(windows.itertuples(index=False), start=1):
        print(f"[{i}/{len(windows)}] {row.sample_id} ({row.target_intent})")
        clip_path = download_clip(row.season, row.episode, row.clip)
        audio_path = extract_audio(clip_path, row.sample_id)
        frame_dir = extract_frames(clip_path, row.sample_id)
        audio_paths.append(audio_path.relative_to(PROJECT_ROOT).as_posix())
        frame_dirs.append(frame_dir.relative_to(PROJECT_ROOT).as_posix())

    windows["audio_path"] = audio_paths
    windows["frame_dir"] = frame_dirs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "index.csv"
    windows.to_csv(index_path, index=False)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote {len(windows)} rows to {index_path.relative_to(PROJECT_ROOT)}")
    print(f"Episodes: {len(SELECTED_EPISODES)}, context window size: {CONTEXT_SIZE}")
    print("Next: python src/extract_dialogue_embeddings.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/generate_dialogue_data.py
```

Expected: a banner, "Loaded N rows across 10 episodes" (N should be 57+44+42+61+60+53+54+51+50+40 = 512), then 512 numbered lines, then "DONE. Wrote 512 rows to data\dialogue\index.csv". Will take several minutes (real download + ~2,500-3,500 subprocess calls) — expected.

- [ ] **Step 3: Verify the generated files**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd
from pathlib import Path

root = Path('.')
idx = pd.read_csv(root / 'data' / 'dialogue' / 'index.csv')
assert len(idx) == 512, f'expected 512 rows, got {len(idx)}'
assert idx.groupby(['season', 'episode']).ngroups == 10, 'expected 10 episodes'

idx['context_sample_ids'] = idx['context_sample_ids'].fillna('')
n_with_full_context = (idx['context_sample_ids'].str.split(';').str.len() == 4).sum()
n_with_no_context = (idx['context_sample_ids'] == '').sum()
print(f'rows with full 4-utterance context: {n_with_full_context}')
print(f'rows with zero context (episode-first utterance): {n_with_no_context}')
assert n_with_no_context == 10, f'expected exactly 10 episode-first rows (one per episode), got {n_with_no_context}'

# Spot-check one context chain resolves to real prior rows in the same episode.
sample = idx[idx['context_sample_ids'].str.split(';').str.len() == 4].iloc[0]
context_ids = sample['context_sample_ids'].split(';')
for cid in context_ids:
    assert cid in idx['sample_id'].values, f'{cid} not found as its own row'
    ctx_row = idx[idx['sample_id'] == cid].iloc[0]
    assert ctx_row['season'] == sample['season'] and ctx_row['episode'] == sample['episode'], 'context from wrong episode'

audio_path = root / sample['audio_path']
frame_dir = root / sample['frame_dir']
assert audio_path.exists(), f'{audio_path} missing'
frames = sorted(frame_dir.glob('frame_*.jpg'))
assert len(frames) == 5, f'expected 5 frames, got {len(frames)}'

print('All checks passed.')
"
```

Expected: `All checks passed.` If the multiline `-c` invocation fails with `NotImplementedError` on this machine, write the identical script to a temp `.py` file and run that instead.

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/generate_dialogue_data.py` explicitly — not `data/dialogue/` (gitignored, matches the rest of `data/`). Suggested message subject: `feat: add dialogue data acquisition for Milestone 4`.

---

## Task 2: Sequence-level embedding extraction

**Files:**
- Create: `src/extract_dialogue_embeddings.py`

**Interfaces:**
- Consumes: `data/dialogue/index.csv` and its `audio_path`/`frame_dir` files (from Task 1).
- Produces: `data/dialogue/embeddings/{sample_id}.npz` with `audio` (shape `(8, 768)`, NOT pooled to one vector — a subsampled sequence) and `video` (shape `(5, 512)`, NOT averaged — all 5 frame embeddings kept).

- [ ] **Step 1: Write `src/extract_dialogue_embeddings.py`**

```python
"""
Milestone 4: extract SEQUENCE-level frozen encoder embeddings for dialogue
context / cross-modal attention (not pooled to one vector, like M2/M3).
=================================================================
GOAL OF THIS FILE
    Milestone 2 pooled each clip's audio into ONE 768-dim vector (mean over
    all of wav2vec2's timesteps) and each clip's video into ONE 512-dim
    vector (mean over 5 frames) -- fine for a plain classifier, but a
    CROSS-MODAL TRANSFORMER needs something to attend OVER, not a single
    already-collapsed vector. This file keeps a short sequence per
    modality instead of fully pooling it:
      - audio: wav2vec2's per-timestep outputs, subsampled down to a
        fixed 8 steps (adaptive mean-pooling, handles any input length)
      - video: all 5 frame embeddings, kept separate (not averaged)

    Both encoders stay FROZEN (no fine-tuning), same as Milestone 2.

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py    # once, if you haven't already
    python src/extract_dialogue_embeddings.py

    Reuses the same wav2vec2-base/ResNet18 weights already downloaded for
    Milestone 2 -- no new download. Safe to re-run: samples that already
    have a cached embedding are skipped.
"""

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "dialogue" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "dialogue" / "embeddings"

AUDIO_CKPT = "facebook/wav2vec2-base"
AUDIO_SEQ_LEN = 8


def load_audio_encoder():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    model.eval()
    return feature_extractor, model


def load_video_encoder():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def read_wav_as_array(wav_path):
    """Read a mono 16-bit PCM WAV file into a float32 numpy array in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def subsample_to_fixed_length(sequence, target_len):
    """Adaptively mean-pool a (T, D) tensor down to exactly (target_len, D),
    regardless of T -- handles T < target_len, T == target_len, T > target_len."""
    x = sequence.transpose(0, 1).unsqueeze(0)  # (1, D, T)
    pooled = F.adaptive_avg_pool1d(x, target_len)  # (1, D, target_len)
    return pooled.squeeze(0).transpose(0, 1)  # (target_len, D)


def embed_audio(wav_path, feature_extractor, model):
    audio = read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        sequence = outputs.last_hidden_state.squeeze(0)  # (time_steps, 768)
        pooled = subsample_to_fixed_length(sequence, AUDIO_SEQ_LEN)  # (8, 768)
    return pooled.numpy()


def embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.stack(embeddings, axis=0)  # (5, 512)


def main():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the dialogue data first:\n"
            "    python src/generate_dialogue_data.py\n"
        )

    print("=" * 70)
    print("EXTRACT DIALOGUE EMBEDDINGS (sequence-level, frozen wav2vec2-base + ResNet18)")
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
    print("Next: python src/train_crossmodal.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/extract_dialogue_embeddings.py
```

Expected: a banner, loaded-row count, encoder-loading lines, 512 numbered progress lines, "DONE. Wrote embeddings for 512 samples...". CPU inference over 512 clips will take several minutes — expected.

- [ ] **Step 3: Verify the generated embeddings**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import numpy as np
import pandas as pd
from pathlib import Path

root = Path('.')
idx = pd.read_csv(root / 'data' / 'dialogue' / 'index.csv')
emb_dir = root / 'data' / 'dialogue' / 'embeddings'
emb_files = list(emb_dir.glob('*.npz'))
assert len(emb_files) == len(idx), f'expected {len(idx)} embeddings, got {len(emb_files)}'

sample = np.load(emb_dir / f'{idx.iloc[0][\"sample_id\"]}.npz')
assert sample['audio'].shape == (8, 768), f'unexpected audio shape {sample[\"audio\"].shape}'
assert sample['video'].shape == (5, 512), f'unexpected video shape {sample[\"video\"].shape}'
assert not np.isnan(sample['audio']).any()
assert not np.isnan(sample['video']).any()

print('All checks passed.')
"
```

Expected: `All checks passed.` (Use a temp `.py` file if the multiline `-c` invocation fails on this machine.)

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/extract_dialogue_embeddings.py` explicitly — not `data/dialogue/embeddings/` (gitignored). Suggested message subject: `feat: add sequence-level embedding extraction for Milestone 4`.

---

## Task 3: Cross-modal dialogue model

**Files:**
- Create: `src/crossmodal_model.py`

**Interfaces:**
- Produces: `CrossModalDialogueModel(text_dim, audio_dim=768, video_dim=512, hidden_dim=128, num_classes=20, use_cross_attention=True)`, a `torch.nn.Module`. `forward(text_window, audio_window, video_window, padding_mask=None)` where `text_window` is `(batch, WINDOW_SIZE, text_dim)`, `audio_window` is `(batch, WINDOW_SIZE, 8, 768)`, `video_window` is `(batch, WINDOW_SIZE, 5, 512)`, `padding_mask` is `(batch, WINDOW_SIZE)` bool (`True` = ignore this position, matching PyTorch's `src_key_padding_mask` convention) or `None` (no padding). Returns `logits` shape `(batch, num_classes)`. `WINDOW_SIZE = 5` and `CONTEXT_SIZE = 4` are module-level constants.
- The `use_cross_attention=False` constructor flag builds a same-capacity ablation variant (plain mean-pooling instead of cross-attention) for Task 4's honest comparison.

- [ ] **Step 1: Write `src/crossmodal_model.py`**

```python
"""
Milestone 4: cross-modal transformer model for dialogue-aware intent
classification.
=================================================================
GOAL OF THIS FILE
    Milestone 3's MISA used self-attention over 6 FIXED representations
    (3 shared + 3 private) of a single utterance. This file implements
    something different: CROSS-modal attention, where one modality's
    sequence directly attends over another modality's sequence, plus
    DIALOGUE context -- the model sees not just the current utterance but
    up to 4 utterances before it in the same conversation.

    Simplified from the original "Multimodal Transformer" (MulT, Tsai et
    al. 2019): only audio->text and video->text cross-attention are
    implemented (not all 6 directional pairs), since every prior
    milestone found text the strongest single modality by a wide margin.
    See docs/superpowers/specs/2026-08-06-milestone4-crossmodal-dialogue-design.md
    for the full reasoning.

TERMS YOU'LL SEE
    - cross-attention : one sequence (the "query") looks at another
      sequence (the "key"/"value") to decide what to pay attention to --
      unlike self-attention, where a sequence looks at itself
    - query / key / value : the three inputs attention mechanisms use --
      informally, "what am I looking for" (query), "what's available to
      match against" (key), and "what do I actually retrieve" (value)
    - dialogue window : the current (target) utterance plus up to 4
      utterances immediately before it in the same conversation
    - padding mask : marks positions in a batch that aren't real data
      (e.g. an episode's first utterance has no 4th/3rd/2nd/1st prior
      utterance to fill a full window) so attention ignores them instead
      of treating placeholder zeros as real content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE = 4
WINDOW_SIZE = CONTEXT_SIZE + 1  # +1 for the target utterance itself


class CrossModalDialogueModel(nn.Module):
    def __init__(self, text_dim, audio_dim=768, video_dim=512, hidden_dim=128,
                 num_classes=20, use_cross_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.audio_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.video_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        dialogue_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=2 * hidden_dim, batch_first=True
        )
        self.dialogue_self_attn = nn.TransformerEncoder(dialogue_layer, num_layers=1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_utterance(self, text_feat, audio_seq, video_seq):
        """
        text_feat:  (batch, text_dim)
        audio_seq:  (batch, 8, audio_dim)
        video_seq:  (batch, 5, video_dim)
        Returns: (batch, hidden_dim) -- one enriched representation per utterance.
        """
        text_repr = self.text_proj(text_feat).unsqueeze(1)  # (batch, 1, H)
        audio_repr = self.audio_proj(audio_seq)              # (batch, 8, H)
        video_repr = self.video_proj(video_seq)               # (batch, 5, H)

        if self.use_cross_attention:
            audio_enriched, _ = self.audio_to_text_attn(text_repr, audio_repr, audio_repr)  # (batch, 1, H)
            video_enriched, _ = self.video_to_text_attn(text_repr, video_repr, video_repr)  # (batch, 1, H)
        else:
            # Ablation: same information available to the model, but combined
            # via plain mean-pooling instead of cross-attention -- isolates
            # what cross-attention specifically contributes, on top of the
            # same encoders/dialogue self-attention/classifier.
            audio_enriched = audio_repr.mean(dim=1, keepdim=True)
            video_enriched = video_repr.mean(dim=1, keepdim=True)

        enriched = text_repr + audio_enriched + video_enriched  # (batch, 1, H)
        return enriched.squeeze(1)  # (batch, H)

    def forward(self, text_window, audio_window, video_window, padding_mask=None):
        """
        text_window:  (batch, WINDOW_SIZE, text_dim)
        audio_window: (batch, WINDOW_SIZE, 8, audio_dim)
        video_window: (batch, WINDOW_SIZE, 5, video_dim)
        padding_mask: (batch, WINDOW_SIZE) bool, True = padding position to
                       ignore. The target position (last) is never padding.
                       None means no padding (every position is real).
        (Positions 0..WINDOW_SIZE-2 are context, oldest-first; position
        WINDOW_SIZE-1 is the target utterance.)
        Returns: logits (batch, num_classes)
        """
        batch, window, _ = text_window.shape
        utterance_reprs = []
        for pos in range(window):
            repr_pos = self.encode_utterance(
                text_window[:, pos, :], audio_window[:, pos, :, :], video_window[:, pos, :, :]
            )
            utterance_reprs.append(repr_pos)
        sequence = torch.stack(utterance_reprs, dim=1)  # (batch, window, H)

        fused_sequence = self.dialogue_self_attn(sequence, src_key_padding_mask=padding_mask)  # (batch, window, H)
        target_repr = fused_sequence[:, -1, :]  # last position = target utterance

        logits = self.classifier(target_repr)
        return logits


if __name__ == "__main__":
    # Smoke test: run with `python src/crossmodal_model.py`
    torch.manual_seed(0)
    batch, text_dim, num_classes = 4, 1400, 20
    model = CrossModalDialogueModel(text_dim=text_dim, num_classes=num_classes)

    text_window = torch.randn(batch, WINDOW_SIZE, text_dim)
    audio_window = torch.randn(batch, WINDOW_SIZE, 8, 768)
    video_window = torch.randn(batch, WINDOW_SIZE, 5, 512)
    labels = torch.randint(0, num_classes, (batch,))

    logits = model(text_window, audio_window, video_window)
    assert logits.shape == (batch, num_classes), f"unexpected logits shape {logits.shape}"
    loss = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss), "loss is not finite"

    # Also smoke-test the padding mask path and the no-cross-attention ablation.
    padding_mask = torch.zeros(batch, WINDOW_SIZE, dtype=torch.bool)
    padding_mask[:, :2] = True  # first 2 context positions are padding for this batch
    logits_padded = model(text_window, audio_window, video_window, padding_mask=padding_mask)
    assert logits_padded.shape == (batch, num_classes), "padding-mask path shape mismatch"
    assert torch.isfinite(logits_padded).all(), "padding-mask path produced non-finite logits"

    ablation_model = CrossModalDialogueModel(text_dim=text_dim, num_classes=num_classes, use_cross_attention=False)
    logits_ablation = ablation_model(text_window, audio_window, video_window)
    assert logits_ablation.shape == (batch, num_classes), "ablation model shape mismatch"

    print("Smoke test passed.")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"padding-mask path: OK, ablation (use_cross_attention=False) path: OK")
```

- [ ] **Step 2: Run the smoke test**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/crossmodal_model.py
```

Expected: `Smoke test passed.` with `logits shape: (4, 20)`, a finite loss value, and confirmation both the padding-mask path and the ablation path work.

- [ ] **Step 3: Commit**

Use the `commit` skill. Stage `src/crossmodal_model.py` explicitly. Suggested message subject: `feat: add cross-modal dialogue model for Milestone 4`.

---

## Task 4: Training script (episode-level split, honest ablation comparison)

**Files:**
- Create: `src/train_crossmodal.py`

**Interfaces:**
- Consumes: `data/dialogue/index.csv` (Task 1), `data/dialogue/embeddings/*.npz` (Task 2), `CrossModalDialogueModel`/`WINDOW_SIZE`/`CONTEXT_SIZE` from `src/crossmodal_model.py` (Task 3).
- Produces: `results/crossmodal_predictions.csv` (`text, true, predicted`) and `results/crossmodal_confusion_matrix.csv`.

- [ ] **Step 1: Write `src/train_crossmodal.py`**

```python
"""
Milestone 4: train and evaluate the cross-modal dialogue model on real
MIntRec dialogue data.
=================================================================
GOAL OF THIS FILE
    Same load -> split -> train -> evaluate shape as Milestone 3, but:
      - the split is by EPISODE, not by individual utterance (a target
        utterance's context must stay in the same split as the target,
        or the split would leak information across train/val/test)
      - alongside the full cross-modal model, this script ALSO trains and
        reports a same-capacity ABLATION (identical architecture, but
        cross-attention replaced with plain mean-pooling) so this
        milestone doesn't repeat Milestone 3's mistake of crediting an
        accuracy change to the wrong mechanism without checking

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py        # once
    python src/extract_dialogue_embeddings.py    # once
    python src/train_crossmodal.py

    Trains two small models (full + ablation) on ~350-400 rows each --
    should finish in well under a minute total on CPU.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from crossmodal_model import CrossModalDialogueModel, WINDOW_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "dialogue" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "dialogue" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

HIDDEN_DIM = 128
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
SEED = 0

AUDIO_SEQ_LEN = 8
N_FRAMES = 5

# Episodes assigned to each split -- by EPISODE, not by utterance, so a
# target utterance's context never crosses into a different split than
# the target itself. 7 train / 1 val / 2 test, seeded selection (not
# stratified -- stratifying 10 groups by which of 20 classes they contain
# isn't practically meaningful with groups this small).
TRAIN_EPISODES = [
    ("S04", "E16"), ("S04", "E04"), ("S04", "E01"), ("S05", "E19"),
    ("S05", "E18"), ("S06", "E03"), ("S06", "E04"),
]
VAL_EPISODES = [("S06", "E02")]
TEST_EPISODES = [("S05", "E20"), ("S06", "E01")]


def load_data():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the dialogue data first:\n"
            "    python src/generate_dialogue_data.py\n"
        )
    if not EMBEDDINGS_DIR.exists() or not any(EMBEDDINGS_DIR.glob("*.npz")):
        raise SystemExit(
            f"\nNo embeddings found in {EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}.\n"
            "Extract embeddings first:\n"
            "    python src/extract_dialogue_embeddings.py\n"
        )
    print("=" * 70)
    print("STEP 1  LOAD THE DIALOGUE DATA")
    print("=" * 70)
    df = pd.read_csv(INDEX_PATH)
    df["context_sample_ids"] = df["context_sample_ids"].fillna("")
    print(f"Loaded {len(df)} rows across {df.groupby(['season', 'episode']).ngroups} episodes")
    return df


def split_data(df):
    print("\n" + "=" * 70)
    print("STEP 2  SPLIT BY EPISODE (not by utterance -- avoids context leakage)")
    print("=" * 70)

    def episode_in(row, episode_set):
        return (row["season"], row["episode"]) in episode_set

    train = df[df.apply(lambda r: episode_in(r, TRAIN_EPISODES), axis=1)].reset_index(drop=True)
    val = df[df.apply(lambda r: episode_in(r, VAL_EPISODES), axis=1)].reset_index(drop=True)
    test = df[df.apply(lambda r: episode_in(r, TEST_EPISODES), axis=1)].reset_index(drop=True)

    print(f"train: {len(train)} samples across {len(TRAIN_EPISODES)} episodes")
    print(f"val:   {len(val)} samples across {len(VAL_EPISODES)} episode")
    print(f"test:  {len(test)} samples across {len(TEST_EPISODES)} episodes")
    return train, val, test


def load_npz(sample_id):
    data = np.load(EMBEDDINGS_DIR / f"{sample_id}.npz")
    return data["audio"], data["video"]  # (8, 768), (5, 512)


def build_window_tensors(df, tfidf, fit):
    """
    For every row, build text/audio/video window tensors and a padding
    mask. Context positions (0..WINDOW_SIZE-2) are LEFT-padded (earliest
    real context goes as far left as fits); position WINDOW_SIZE-1 is
    always the target (never padding). Context utterances always come
    from the SAME episode as their target, and episodes are entirely
    within one split, so every context lookup resolves within this same
    dataframe -- no cross-split lookup is possible or needed.
    """
    if fit:
        text_matrix = tfidf.fit_transform(df["target_text"]).toarray().astype(np.float32)
    else:
        text_matrix = tfidf.transform(df["target_text"]).toarray().astype(np.float32)
    text_dim = text_matrix.shape[1]
    text_by_id = {sid: text_matrix[i] for i, sid in enumerate(df["sample_id"])}

    n = len(df)
    text_window = np.zeros((n, WINDOW_SIZE, text_dim), dtype=np.float32)
    audio_window = np.zeros((n, WINDOW_SIZE, AUDIO_SEQ_LEN, 768), dtype=np.float32)
    video_window = np.zeros((n, WINDOW_SIZE, N_FRAMES, 512), dtype=np.float32)
    padding_mask = np.ones((n, WINDOW_SIZE), dtype=bool)  # start all-padding, fill in real positions

    for row_idx, row in enumerate(df.itertuples(index=False)):
        context_ids = [c for c in row.context_sample_ids.split(";") if c]
        n_context = len(context_ids)
        start = WINDOW_SIZE - 1 - n_context  # left-pad: earliest real context goes here

        for j, sid in enumerate(context_ids):
            pos = start + j
            padding_mask[row_idx, pos] = False
            a, v = load_npz(sid)
            audio_window[row_idx, pos] = a
            video_window[row_idx, pos] = v
            text_window[row_idx, pos] = text_by_id[sid]

        padding_mask[row_idx, WINDOW_SIZE - 1] = False
        a, v = load_npz(row.sample_id)
        audio_window[row_idx, WINDOW_SIZE - 1] = a
        video_window[row_idx, WINDOW_SIZE - 1] = v
        text_window[row_idx, WINDOW_SIZE - 1] = text_by_id[row.sample_id]

    return text_window, audio_window, video_window, padding_mask, text_dim


def to_tensors(text, audio, video, mask, labels, label2id):
    return (
        torch.from_numpy(text), torch.from_numpy(audio), torch.from_numpy(video),
        torch.from_numpy(mask),
        torch.tensor([label2id[l] for l in labels], dtype=torch.long),
    )


def run_epoch(model, optimizer, text, audio, video, mask, labels, batch_size, train_mode):
    model.train(train_mode)
    n = text.shape[0]
    indices = torch.randperm(n) if train_mode else torch.arange(n)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        bt, ba, bv, bm, bl = text[batch_idx], audio[batch_idx], video[batch_idx], mask[batch_idx], labels[batch_idx]

        if train_mode:
            optimizer.zero_grad()
            logits = model(bt, ba, bv, padding_mask=bm)
            loss = torch.nn.functional.cross_entropy(logits, bl)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(bt, ba, bv, padding_mask=bm)
                loss = torch.nn.functional.cross_entropy(logits, bl)

        total_loss += loss.item() * len(batch_idx)
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(bl.tolist())

    avg_loss = total_loss / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_model(text_dim, train_tensors, val_tensors, use_cross_attention, label):
    print("\n" + "=" * 70)
    print(f"STEP 3  TRAIN: {label}")
    print("=" * 70)
    text_train, audio_train, video_train, mask_train, labels_train = train_tensors
    text_val, audio_val, video_val, mask_val, labels_val = val_tensors

    torch.manual_seed(SEED)
    model = CrossModalDialogueModel(
        text_dim=text_dim, hidden_dim=HIDDEN_DIM, num_classes=20, use_cross_attention=use_cross_attention
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(
            model, optimizer, text_train, audio_train, video_train, mask_train, labels_train, BATCH_SIZE, True
        )
        val_loss, val_f1 = run_epoch(
            model, optimizer, text_val, audio_val, video_val, mask_val, labels_val, BATCH_SIZE, False
        )
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1, best_state, epochs_without_improvement = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improvement += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            marker = " *" if improved else ""
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_f1={train_f1:.3f}  "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.3f}{marker}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break
    else:
        print(f"\nReached max epochs ({MAX_EPOCHS}) without early stopping.")

    model.load_state_dict(best_state)
    print(f"Restored best checkpoint (val macro-F1={best_val_f1:.3f}).")
    return model


def evaluate(model, test_tensors, id2label, test_df, label):
    print("\n" + "=" * 70)
    print(f"STEP 4  EVALUATE: {label}")
    print("=" * 70)
    text_test, audio_test, video_test, mask_test, labels_test = test_tensors

    model.eval()
    with torch.no_grad():
        logits = model(text_test, audio_test, video_test, padding_mask=mask_test)
    preds_id = logits.argmax(dim=1).tolist()
    preds = [id2label[p] for p in preds_id]
    truth = [id2label[l] for l in labels_test.tolist()]

    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro", zero_division=0)
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(truth, preds, zero_division=0))

    return acc, macro_f1, truth, preds


def main():
    df = load_data()
    train, val, test = split_data(df)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    text_train, audio_train, video_train, mask_train, text_dim = build_window_tensors(train, tfidf, fit=True)
    text_val, audio_val, video_val, mask_val, _ = build_window_tensors(val, tfidf, fit=False)
    text_test, audio_test, video_test, mask_test, _ = build_window_tensors(test, tfidf, fit=False)
    print(f"\nText feature dim: {text_dim}")

    labels_sorted = sorted(df["target_intent"].unique())
    label2id = {name: i for i, name in enumerate(labels_sorted)}
    id2label = {i: name for name, i in label2id.items()}

    train_tensors = to_tensors(text_train, audio_train, video_train, mask_train, train["target_intent"].tolist(), label2id)
    val_tensors = to_tensors(text_val, audio_val, video_val, mask_val, val["target_intent"].tolist(), label2id)
    test_tensors = to_tensors(text_test, audio_test, video_test, mask_test, test["target_intent"].tolist(), label2id)

    # Full model: cross-modal attention + dialogue context.
    full_model = train_model(text_dim, train_tensors, val_tensors, use_cross_attention=True,
                              label="full cross-modal model")
    full_acc, full_f1, truth, preds = evaluate(full_model, test_tensors, id2label, test,
                                                label="full cross-modal model")

    # Ablation: same architecture, cross-attention replaced with mean-pooling.
    ablation_model = train_model(text_dim, train_tensors, val_tensors, use_cross_attention=False,
                                  label="ablation (no cross-attention, mean-pool instead)")
    ablation_acc, ablation_f1, _, _ = evaluate(ablation_model, test_tensors, id2label, test,
                                                label="ablation (no cross-attention)")

    RESULTS_DIR.mkdir(exist_ok=True)
    labels_present = sorted(set(truth) | set(preds))
    cm = confusion_matrix(truth, preds, labels=labels_present)
    cm_path = RESULTS_DIR / "crossmodal_confusion_matrix.csv"
    pd.DataFrame(cm, index=labels_present, columns=labels_present).to_csv(cm_path)

    out = RESULTS_DIR / "crossmodal_predictions.csv"
    pd.DataFrame({"text": test["target_text"].tolist(), "true": truth, "predicted": preds}).to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("DONE. Compare against every prior milestone on this project's real data:")
    print(f"  Full cross-modal + dialogue context (M4): accuracy {full_acc:.3f}, macro-F1 {full_f1:.3f}")
    print(f"  Ablation, no cross-attention (M4):         accuracy {ablation_acc:.3f}, macro-F1 {ablation_f1:.3f}")
    print("  MISA (M3, 300-clip stratified sample):     accuracy 0.333, macro-F1 0.295")
    print("  Concatenation fusion (M2, same sample):    accuracy 0.117, macro-F1 0.096")
    print("Note: M4 uses a DIFFERENT 512-clip dialogue dataset (10 full episodes) than")
    print("M2/M3's 300-clip stratified sample, so these are directional comparisons,")
    print("not a controlled like-for-like -- the full-vs-ablation comparison above,")
    print("on the identical data/split, is the one apples-to-apples number here.")
    print(f"Saved predictions to {out.relative_to(PROJECT_ROOT)}")
    print(f"Saved confusion matrix to {cm_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/train_crossmodal.py
```

Expected: STEP 1-2 banners, text feature dim line, two full train/evaluate cycles (full model, then ablation), each with per-epoch progress and early-stopping/max-epochs messages, a closing comparison block with both real numbers and the honest caveat that M4's data differs from M2/M3's. Should finish in well under a minute for both models combined — if not, note the deviation in your report.

- [ ] **Step 3: Verify results are self-consistent**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd
from sklearn.metrics import accuracy_score

preds = pd.read_csv('results/crossmodal_predictions.csv')
assert list(preds.columns) == ['text', 'true', 'predicted']
acc = accuracy_score(preds['true'], preds['predicted'])
print(f'test rows: {len(preds)}')
print(f'accuracy from saved predictions: {acc:.3f}')

cm = pd.read_csv('results/crossmodal_confusion_matrix.csv', index_col=0)
assert cm.values.sum() == len(preds), f'confusion matrix should sum to {len(preds)}, got {cm.values.sum()}'
print('OK: results files are internally consistent.')
"
```

Expected: `OK: results files are internally consistent.` No accuracy-floor assertion — report whatever the real numbers are, for both the full model and the ablation.

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/train_crossmodal.py` explicitly — not `results/` (gitignored). Suggested message subject: `feat: add cross-modal dialogue training script for Milestone 4`.

---

## Task 5: Tutorial notebook

**Files:**
- Create: `notebooks/milestone4_crossmodal_dialogue.ipynb`
- Create (scratchpad only, not committed): a Python builder script filling the notebook's cells, same throwaway pattern used for every prior notebook.

**Interfaces:**
- Consumes: `data/dialogue/index.csv` (Task 1), `data/dialogue/embeddings/*.npz` (Task 2), imports `CrossModalDialogueModel`/`WINDOW_SIZE` from `src/crossmodal_model.py` (Task 3, via `sys.path.insert`, same pattern as Milestone 3's notebook) rather than redefining the architecture inline.

- [ ] **Step 1: Scaffold the notebook from the tutorial template**

```bash
python "C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py" --kind tutorial --title "Milestone 4 - Cross-Modal Dialogue" --out "notebooks/milestone4_crossmodal_dialogue.ipynb"
```

Expected: `Wrote ...notebooks\milestone4_crossmodal_dialogue.ipynb using kind=tutorial.`

- [ ] **Step 2: Write the cell-filling builder script**

Save to the scratchpad (e.g. `build_m4_crossmodal_notebook.py`). Mirror `src/train_crossmodal.py`'s logic cell-by-cell (imports, `load_data`, `split_data`, `build_window_tensors`, `to_tensors`, `run_epoch`, the two `train_model` calls for full + ablation, `evaluate` for both, the comparison). Structure the notebook with these markdown sections, in order, matching every prior notebook's established tutorial shape (audience/prerequisites/learning-goals intro; outline; numbered steps; a training-curve plot for BOTH the full model and the ablation, side by side, extending Milestone 3's single-model training-curve idea; a confusion-matrix heatmap; a comparison table against every prior milestone with the same honest "different dataset, directional comparison" caveat that's in `src/train_crossmodal.py`'s closing print; a "look at the mistakes" section; exercises — e.g. "try `CONTEXT_SIZE = 2` instead of 4 in `src/crossmodal_model.py`, does less context help or hurt?", "read a few dialogue windows in `data/dialogue/index.csv` and judge for yourself whether clip-number order looks chronological, per the design spec's documented assumption"; and a pitfalls/extensions section that explicitly names the episode-level-split rationale (context leakage) as something a reader could get wrong on their own future projects, plus points to Milestone 5 as where a proper controlled ablation across multiple seeds belongs, per the lesson from Milestone 3's final review.

Explain the new concepts plainly before using them: cross-attention vs. self-attention (contrast with Milestone 3's self-attention-only fusion), what a padding mask is and why it's needed here specifically (episodes' first few utterances don't have 4 real predecessors), and why the split is by episode rather than by row (contrast with every single prior milestone's utterance-level split — this is worth a dedicated markdown cell since it's a real, easy-to-get-wrong methodological point).

- [ ] **Step 3: Run the builder script**

```bash
python <path-to-scratchpad>/build_m4_crossmodal_notebook.py
```

Expected: confirmation of how many cells were written.

- [ ] **Step 4: Execute the notebook top-to-bottom**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi jupyter nbconvert --to notebook --execute --inplace "notebooks/milestone4_crossmodal_dialogue.ipynb"
```

Expected: successful write with no Python traceback (usual symlink/TCP-kernel warnings are expected and harmless). Trains two small models for real — allow it to actually finish (should be under a minute total) rather than assuming success from a background process.

- [ ] **Step 5: Verify no errors in the executed notebook**

```bash
python -c "
import json
nb = json.load(open('notebooks/milestone4_crossmodal_dialogue.ipynb', encoding='utf-8'))
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

Use the `commit` skill. Stage `notebooks/milestone4_crossmodal_dialogue.ipynb` only (the builder script stays in the scratchpad). Suggested message subject: `feat: add Milestone 4 cross-modal dialogue notebook`.

---

## Final check

After Task 5, read back the real accuracy/macro-F1 from Task 4's run (both the full model and the ablation) and the notebook's comparison. This plan does not include a `CLAUDE.md` update step — per this project's established pattern, that's a controller-level step, done after this plan's final review is clean, using the real results, then proceeding to Milestone 5 without waiting for confirmation (per the autonomous-execution authorization covering M3-M5).
