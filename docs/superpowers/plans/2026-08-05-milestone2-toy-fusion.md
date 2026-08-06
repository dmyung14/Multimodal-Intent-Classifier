# Milestone 2 Toy Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the text+audio+video concatenation-fusion pipeline works end to end, on a synthetic zero-download dataset, before touching any real audio/video data.

**Architecture:** A one-time seeded generator script builds fake audio (`.npy` sine waves) and video (`.npy` RGB frames) files from the existing `data/sample_intents.csv`, with signal tied to each of its 8 intent labels. A training script loads all three modalities, extracts small hand-written feature vectors from each (TF-IDF for text, FFT stats for audio, color stats for video), concatenates them, scales them, and trains/evaluates the same `LogisticRegression` classifier used in Milestones 0 and 1. A tutorial notebook runs the same steps narrated, with a confusion-matrix heatmap and a comparison against the Milestone 0 text-only baseline.

**Tech Stack:** Python 3.11 in the existing `mmi` conda environment (pandas, numpy, scikit-learn, matplotlib, jupyter — all already installed, no new packages).

## Global Constraints

- Zero new package installs, zero downloads — spec requires this increment to use only already-installed libraries (numpy, pandas, scikit-learn, matplotlib).
- Reuse the existing 8 intents / 96 rows from `data/sample_intents.csv` — no new text is invented.
- Audio/video signal must be deliberately tied to the intent label (with random jitter), not pure noise, per the approved spec.
- Generation must be deterministic (seeded) — re-running `generate_toy_multimodal.py` must reproduce identical files.
- Follow the established code patterns in `src/train_text_only.py` and `src/train_mintrec.py`: module docstring explaining purpose/how-to-run/terms in plain language, `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, functions named `load_data()`/`split_data()`/`evaluate()`, printed step banners (`"=" * 70`), stratified `train_test_split` with `random_state=42` matching M0/M1 exactly (same split proportions: `test_size=0.20` then `test_size=0.1875`).
- Follow the established notebook pattern: scaffold via `C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py --kind tutorial`, fill cells with a throwaway Python builder script (not committed — lives in the scratchpad, like the M0/M1 notebooks were built), then execute with `nbconvert --to notebook --execute --inplace` before committing.
- Invoke conda via the full path — it is not on PATH: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi <command>`.
- No pytest suite exists in this project (M0/M1 have none either) — verification is "run the script/notebook, read the printed output, check the generated files," matching the approved spec's own Testing/Validation section. Don't introduce a test framework here.
- This project uses the `commit` skill (secret scan, draft message, user approval) for every commit, not raw `git commit -m` without review — follow that flow for each task's commit step.

---

## Task 1: Synthetic data generator

**Files:**
- Create: `src/generate_toy_multimodal.py`

**Interfaces:**
- Produces: `data/toy_multimodal/index.csv` with columns `sample_id, text, intent, audio_path, video_path` (96 rows). `audio_path`/`video_path` are POSIX-style paths relative to the project root (e.g. `data/toy_multimodal/audio/0000.npy`).
- Produces: `data/toy_multimodal/audio/<sample_id>.npy` — 1D float32 array, shape `(4000,)` (0.5s at 8000Hz), one sine wave per row.
- Produces: `data/toy_multimodal/video/<sample_id>.npy` — uint8 array, shape `(16, 16, 3)`, one RGB frame per row.
- Constants later tasks depend on: `SAMPLE_RATE = 8000` (Task 2's audio feature extractor needs this to match).

- [ ] **Step 1: Write `src/generate_toy_multimodal.py`**

```python
"""
Milestone 2, increment 1: generate a synthetic, zero-download multimodal
toy dataset from the existing text-only sample.
=================================================================
GOAL OF THIS FILE
    Milestone 2 is about fusing text + audio + video. Before touching any
    real audio/video data (which lives in an 800MB+ archive we haven't
    downloaded yet), we manufacture tiny FAKE audio and video "clips" for
    each of the 96 sentences already in data/sample_intents.csv, so we can
    build and sanity-check the fusion pipeline in seconds, offline.

    The fake audio is a short sine wave; the fake video is a small solid-ish
    color image. Neither is "real" data -- but each one's frequency/color is
    deliberately tied to its intent label (with random jitter), so a
    classifier CAN learn something from them. That lets the next script
    show fusion actually helping, not just "the code runs."

HOW TO RUN
    conda activate mmi
    python src/generate_toy_multimodal.py

    Finishes in under a second and writes files under data/toy_multimodal/.
    Re-running it is safe -- it's seeded, so it always produces the exact
    same files.

TERMS YOU'LL SEE
    - sine wave : a smooth, single-pitch sound wave; the simplest way to
                  represent a tone as numbers (amplitude at each moment
                  in time)
    - RGB       : each pixel of a color image stored as three numbers
                  (Red, Green, Blue), each 0-255
    - seed      : a fixed starting point for "random" numbers, so
                  re-running the same code produces identical output
                  every time
"""

import colorsys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_INTENTS_PATH = PROJECT_ROOT / "data" / "sample_intents.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "toy_multimodal"

SEED = 0
SAMPLE_RATE = 8000       # audio samples per second. Real speech models use
                          # 16000+; 8000 is plenty for a fake tone and keeps
                          # files tiny. src/train_multimodal_toy.py's audio
                          # feature extractor must use this same value.
AUDIO_DURATION_S = 0.5    # each fake clip is half a second long
FRAME_SIZE = 16           # each fake video frame is 16x16 pixels

INTENTS = ["greeting", "thank", "apologize", "agree", "disagree", "complain", "request", "reassure"]
# One base audio frequency per intent (Hz), evenly spaced 100Hz apart so
# they're easy for a classifier (and for us, reading the numbers) to tell apart.
BASE_FREQUENCIES = {intent: 200 + 100 * i for i, intent in enumerate(INTENTS)}


def intent_base_color(intent):
    """Evenly space each intent around the color wheel so colors are distinct."""
    hue = INTENTS.index(intent) / len(INTENTS)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return np.array([r, g, b]) * 255


def make_audio_clip(intent, rng):
    """A short sine wave whose frequency is tied to the intent, plus jitter."""
    freq = BASE_FREQUENCIES[intent] + rng.uniform(-15, 15)
    t = np.linspace(0, AUDIO_DURATION_S, int(SAMPLE_RATE * AUDIO_DURATION_S), endpoint=False)
    amplitude = rng.uniform(0.7, 1.0)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    noise = rng.normal(0, 0.02, size=wave.shape)  # a little noise so it's not a perfect tone
    return (wave + noise).astype(np.float32)


def make_video_frame(intent, rng):
    """A small RGB image whose average color is tied to the intent, plus jitter."""
    base_color = intent_base_color(intent)
    frame = np.tile(base_color, (FRAME_SIZE, FRAME_SIZE, 1)).astype(np.float32)
    noise = rng.normal(0, 12, size=frame.shape)
    frame = np.clip(frame + noise, 0, 255)
    return frame.astype(np.uint8)


def main():
    print("=" * 70)
    print("GENERATE TOY MULTIMODAL DATA")
    print("=" * 70)

    df = pd.read_csv(SAMPLE_INTENTS_PATH)
    print(f"Loaded {len(df)} rows from {SAMPLE_INTENTS_PATH.name}")

    audio_dir = OUTPUT_DIR / "audio"
    video_dir = OUTPUT_DIR / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for i, row in df.iterrows():
        sample_id = f"{i:04d}"
        audio_path = audio_dir / f"{sample_id}.npy"
        video_path = video_dir / f"{sample_id}.npy"

        np.save(audio_path, make_audio_clip(row["intent"], rng))
        np.save(video_path, make_video_frame(row["intent"], rng))

        rows.append(
            {
                "sample_id": sample_id,
                "text": row["text"],
                "intent": row["intent"],
                "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                "video_path": video_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    index_df = pd.DataFrame(rows)
    index_path = OUTPUT_DIR / "index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"Wrote {len(index_df)} audio clips to {audio_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(index_df)} video frames to {video_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote index to {index_path.relative_to(PROJECT_ROOT)}")
    print("\nDone. This is fake data standing in for real audio/video -- see")
    print("docs/superpowers/specs/2026-08-05-milestone2-toy-fusion-design.md")
    print("for why, and src/train_multimodal_toy.py for how it's used.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/generate_toy_multimodal.py
```

Expected output: a banner, "Loaded 96 rows from sample_intents.csv", "Wrote 96 audio clips to data\toy_multimodal\audio", "Wrote 96 video frames to data\toy_multimodal\video", "Wrote index to data\toy_multimodal\index.csv", then the closing note. Finishes in under 2 seconds.

- [ ] **Step 3: Verify the generated files**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import numpy as np
import pandas as pd
from pathlib import Path

root = Path('.')
idx = pd.read_csv(root / 'data' / 'toy_multimodal' / 'index.csv')
assert len(idx) == 96, f'expected 96 rows, got {len(idx)}'
assert list(idx.columns) == ['sample_id', 'text', 'intent', 'audio_path', 'video_path']

audio_files = list((root / 'data' / 'toy_multimodal' / 'audio').glob('*.npy'))
video_files = list((root / 'data' / 'toy_multimodal' / 'video').glob('*.npy'))
assert len(audio_files) == 96, f'expected 96 audio files, got {len(audio_files)}'
assert len(video_files) == 96, f'expected 96 video files, got {len(video_files)}'

clip = np.load(root / idx.iloc[0]['audio_path'])
frame = np.load(root / idx.iloc[0]['video_path'])
assert clip.shape == (4000,), f'unexpected audio shape {clip.shape}'
assert frame.shape == (16, 16, 3), f'unexpected video shape {frame.shape}'
assert frame.dtype == np.uint8

print('All checks passed.')
print(idx.head(3).to_string(index=False))
"
```

Expected: `All checks passed.` followed by a 3-row preview.

- [ ] **Step 4: Verify determinism (re-running produces identical files)**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import hashlib
from pathlib import Path
p = Path('data/toy_multimodal/audio/0000.npy')
print('before:', hashlib.md5(p.read_bytes()).hexdigest())
"
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/generate_toy_multimodal.py
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import hashlib
from pathlib import Path
p = Path('data/toy_multimodal/audio/0000.npy')
print('after: ', hashlib.md5(p.read_bytes()).hexdigest())
"
```

Expected: the two printed hashes are identical.

- [ ] **Step 5: Commit**

Use the `commit` skill (scans for secrets, drafts message, asks for approval) rather than a raw `git commit`. Stage `src/generate_toy_multimodal.py` explicitly — do not stage `data/toy_multimodal/` (it's gitignored, matching the rest of `data/`, and generated files shouldn't be committed). Suggested message subject: `feat: add toy multimodal data generator for Milestone 2`.

---

## Task 2: Fusion training script

**Files:**
- Create: `src/train_multimodal_toy.py`

**Interfaces:**
- Consumes: `data/toy_multimodal/index.csv` and the `.npy` files it points to (from Task 1). Consumes `SAMPLE_RATE = 8000` (must match Task 1's generator).
- Produces: `results/toy_multimodal_predictions.csv` with columns `text, true, predicted`.
- Produces console output: `STEP 1` through `STEP 4` banners, a "Fused feature vector size" line, `Accuracy` and `Macro-F1` lines, a per-class `classification_report`, and an 8x8 ASCII confusion matrix (8 classes fits a terminal, unlike M1's 20).

- [ ] **Step 1: Write `src/train_multimodal_toy.py`**

```python
"""
Milestone 2, increment 1: fuse text + (synthetic) audio + (synthetic) video.
=================================================================
GOAL OF THIS FILE
    Same load -> split -> train -> evaluate loop as Milestone 0, but now
    with THREE modalities fused together instead of one:

        text  ---TF-IDF--->        \
        audio ---FFT stats--->      >--concatenate--> scale --> classifier
        video ---color stats--->   /

    This runs on the FAKE data built by generate_toy_multimodal.py (run
    that first). The point isn't realism -- it's proving the fusion
    mechanics (loading three modalities, extracting features from each,
    concatenating them into one vector, training on it) work correctly,
    before we ever touch real audio/video data.

HOW TO RUN
    conda activate mmi
    python src/generate_toy_multimodal.py   # once, if you haven't already
    python src/train_multimodal_toy.py

    Finishes in a couple of seconds, CPU only.

TERMS YOU'LL SEE (new ones, beyond what Milestone 0 introduced)
    - FFT (Fast Fourier Transform): a way to take a sound wave (numbers
      over time) and find out which pitches/frequencies are present in it.
      We use it to recover each fake clip's dominant frequency.
    - concatenation fusion: the simplest way to combine multiple
      modalities -- extract a feature vector from each one separately,
      then glue them end to end into one longer vector before the
      classifier ever sees them.
    - feature scaling: TF-IDF values are tiny (0-1ish), our fake audio
      frequency is in the hundreds, and RGB values go up to 255. Fed in
      raw, the biggest-magnitude numbers would unfairly dominate what the
      classifier learns. StandardScaler rescales every feature to have
      mean 0 and standard deviation 1 first, so no modality overpowers
      the others just because its numbers happen to be bigger.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "toy_multimodal" / "index.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLE_RATE = 8000  # must match generate_toy_multimodal.py's SAMPLE_RATE


def load_data():
    """Read the toy multimodal index and show what we're working with."""
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the toy multimodal data first:\n"
            "    python src/generate_toy_multimodal.py\n"
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
    """Same stratified train/val/test split as Milestone 0."""
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


def extract_audio_features(clip):
    """
    FFT-based stand-in for a real audio encoder: dominant frequency (which
    pitch is loudest) and RMS energy (how loud the clip is overall).
    """
    fft_vals = np.fft.rfft(clip)
    fft_freqs = np.fft.rfftfreq(len(clip), d=1 / SAMPLE_RATE)
    dominant_freq = fft_freqs[np.argmax(np.abs(fft_vals))]
    energy = np.sqrt(np.mean(clip ** 2))
    return np.array([dominant_freq, energy])


def extract_video_features(frame):
    """
    Color-stats stand-in for a real video encoder: mean red/green/blue and
    how much the brightness varies across the frame.
    """
    frame = frame.astype(np.float32)
    mean_r, mean_g, mean_b = frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()
    brightness = frame.mean(axis=2)
    brightness_variance = brightness.var()
    return np.array([mean_r, mean_g, mean_b, brightness_variance])


def build_audio_features(df):
    return np.array([extract_audio_features(np.load(PROJECT_ROOT / p)) for p in df["audio_path"]])


def build_video_features(df):
    return np.array([extract_video_features(np.load(PROJECT_ROOT / p)) for p in df["video_path"]])


def fit_transform_features(df):
    """Fit the text vectorizer + scaler on TRAIN, return fused+scaled features."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    text_feats = tfidf.fit_transform(df["text"]).toarray()
    audio_feats = build_audio_features(df)
    video_feats = build_video_features(df)
    fused = np.hstack([text_feats, audio_feats, video_feats])

    scaler = StandardScaler()
    fused_scaled = scaler.fit_transform(fused)
    return fused_scaled, tfidf, scaler, text_feats.shape[1]


def transform_features(df, tfidf, scaler):
    """Apply the already-fitted vectorizer + scaler to VAL/TEST."""
    text_feats = tfidf.transform(df["text"]).toarray()
    audio_feats = build_audio_features(df)
    video_feats = build_video_features(df)
    fused = np.hstack([text_feats, audio_feats, video_feats])
    return scaler.transform(fused)


def train_fused(train):
    """Build the fused feature vectors and train Logistic Regression on them."""
    print("\n" + "=" * 70)
    print("STEP 3  EXTRACT FEATURES FROM ALL THREE MODALITIES AND FUSE")
    print("=" * 70)
    fused_train, tfidf, scaler, n_text_features = fit_transform_features(train)
    print(
        f"Fused feature vector size: {fused_train.shape[1]} "
        f"(text={n_text_features}, audio=2, video=4)"
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(fused_train, train["intent"])
    print("Training done.")
    return tfidf, scaler, clf


def evaluate(tfidf, scaler, clf, test):
    """Judge the fused model on the held-out test set, same metrics as Milestone 0."""
    print("\n" + "=" * 70)
    print("STEP 4  EVALUATE ON THE TEST SET")
    print("=" * 70)

    fused_test = transform_features(test, tfidf, scaler)
    preds = clf.predict(fused_test)
    truth = test["intent"].tolist()

    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro")
    print(f"Accuracy : {acc:.3f}   (share of predictions that were exactly right)")
    print(f"Macro-F1 : {macro_f1:.3f}   (fairer average across all intents)")

    print("\nPer-class report (precision/recall/F1 for each intent):")
    print(classification_report(truth, preds, zero_division=0))

    labels = sorted(test["intent"].unique())
    cm = confusion_matrix(truth, preds, labels=labels)
    print("Confusion matrix (rows = true intent, columns = predicted intent):")
    header = "true\\pred".ljust(12) + "".join(l[:8].ljust(9) for l in labels)
    print(header)
    for name, row in zip(labels, cm):
        print(name[:11].ljust(12) + "".join(str(v).ljust(9) for v in row))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "toy_multimodal_predictions.csv"
    pd.DataFrame({"text": test["text"], "true": truth, "predicted": preds}).to_csv(out, index=False)
    print(f"\nSaved every test prediction to {out.relative_to(PROJECT_ROOT)}")


def main():
    df = load_data()
    train, val, test = split_data(df)
    tfidf, scaler, clf = train_fused(train)
    evaluate(tfidf, scaler, clf, test)

    print("\n" + "=" * 70)
    print("DONE. Compare this accuracy to Milestone 0's text-only 0.550 -- ask Claude Code:")
    print("  - 'Did fusing fake audio/video actually help, or hurt?'")
    print("  - 'What would happen if I skipped StandardScaler here?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/train_multimodal_toy.py
```

Expected: STEP 1-4 banners; a line like `Fused feature vector size: 488 (text=482, audio=2, video=4)` (text feature count should match Milestone 0's 482, since it's fit on the identical 61-row training split); `Accuracy` and `Macro-F1` lines; a classification report; an 8x8 ASCII confusion matrix; `Saved every test prediction to results\toy_multimodal_predictions.csv`. Finishes in a few seconds.

- [ ] **Step 3: Verify accuracy meets the spec's bar and the results file is correct**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd
from sklearn.metrics import accuracy_score

preds = pd.read_csv('results/toy_multimodal_predictions.csv')
assert len(preds) == 20, f'expected 20 test rows, got {len(preds)}'
assert list(preds.columns) == ['text', 'true', 'predicted']

acc = accuracy_score(preds['true'], preds['predicted'])
print(f'accuracy from saved predictions: {acc:.3f}')
assert acc >= 0.550, (
    f'fused accuracy {acc:.3f} is below the M0 text-only baseline (0.550) -- '
    'per the design spec this needs investigating before moving on'
)
print('OK: fused accuracy meets or beats the text-only baseline.')
"
```

Expected: `OK: fused accuracy meets or beats the text-only baseline.` If the assertion fails instead, stop and investigate (e.g. check the `StandardScaler` step, check that audio/video features are actually varying by intent) before proceeding to Task 3 — don't silently continue with a broken fusion result.

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/train_multimodal_toy.py` explicitly (not `results/`, which is gitignored). Suggested message subject: `feat: add fused text+audio+video training script for Milestone 2`.

---

## Task 3: Tutorial notebook

**Files:**
- Create: `notebooks/milestone2_toy_fusion.ipynb`
- Create (scratchpad only, not committed): a Python builder script that fills the notebook's cells — follow the same throwaway-script pattern used for `notebooks/milestone0_text_only.ipynb` and `notebooks/milestone1_mintrec_text.ipynb` (write it under the session scratchpad directory, not under the project).

**Interfaces:**
- Consumes: everything from Tasks 1 and 2 — `data/toy_multimodal/index.csv`, and the same feature-extraction/fusion logic as `src/train_multimodal_toy.py` (the notebook re-implements it in cells, matching how `milestone0_text_only.ipynb` re-implements `train_text_only.py` rather than importing it).
- Produces: an executed notebook with real outputs (printed metrics, a confusion matrix heatmap image, a comparison table) saved in it.

- [ ] **Step 1: Scaffold the notebook from the tutorial template**

```bash
python "C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py" --kind tutorial --title "Milestone 2 - Toy Multimodal Fusion" --out "notebooks/milestone2_toy_fusion.ipynb"
```

Expected: `Wrote ...notebooks\milestone2_toy_fusion.ipynb using kind=tutorial.`

- [ ] **Step 2: Write the cell-filling builder script**

Save this to a scratchpad file (e.g. `build_m2_notebook.py`), adjusting `NB_PATH` if your scratchpad differs:

```python
"""Builds notebooks/milestone2_toy_fusion.ipynb by filling the scaffolded template."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\dbest\Downloads\multimodal-intent-inference\multimodal-intent-inference\notebooks\milestone2_toy_fusion.ipynb")


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
        "# Tutorial: Milestone 2 - Toy Multimodal Fusion",
        "",
        "Audience:",
        "- You've completed Milestones 0 and 1 (text-only, on toy and then real data) and understand the",
        "  load -> split -> train -> evaluate loop plus TF-IDF, accuracy, and macro-F1.",
        "",
        "Prerequisites:",
        "- Run `python src/generate_toy_multimodal.py` first (from a terminal, in the `mmi` environment) --",
        "  this notebook loads the files it creates under `data/toy_multimodal/`.",
        "",
        "Learning goals:",
        "- By the end, you can explain what 'concatenation fusion' means, extract simple features from three",
        "  different modalities, and explain why feature scaling matters when combining them.",
        "",
        "**Why synthetic data:** the real MIntRec audio/video lives in an 800MB+ archive we haven't downloaded",
        "yet (out of scope for this increment -- see `docs/superpowers/specs/2026-08-05-milestone2-toy-fusion-design.md`).",
        "Here we generate small FAKE audio/video files whose frequency/color is deliberately tied to each of the",
        "8 intents, so we can prove the fusion *mechanics* work in seconds, offline, with zero new installs.",
    ),
    md(
        "## Outline",
        "",
        "1. Setup",
        "2. Step 1 - Load the toy multimodal index",
        "3. Step 2 - Split into train / validation / test",
        "4. Step 3 - Extract features from each modality and fuse them",
        "5. Step 4 - Evaluate, and compare against the Milestone 0 text-only baseline",
        "6. Look at the mistakes",
        "7. Exercises",
        "8. Pitfalls and extensions",
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
        "INDEX_PATH = PROJECT_ROOT / \"data\" / \"toy_multimodal\" / \"index.csv\"",
        "RESULTS_DIR = PROJECT_ROOT / \"results\"",
        "SAMPLE_RATE = 8000  # must match src/generate_toy_multimodal.py",
        "INDEX_PATH",
    ),
    md(
        "## Step 1 - Load the toy multimodal index",
        "",
        "`index.csv` (built by `src/generate_toy_multimodal.py`) has one row per sample, pointing at that",
        "sample's fake audio and video files. It doesn't contain the audio/video data itself -- just paths to it,",
        "the same way the real MIntRec dataset's `.tsv` files point at separate video segment files.",
    ),
    code(
        "df = pd.read_csv(INDEX_PATH)",
        "print(f\"Loaded {len(df)} samples from {INDEX_PATH.name}\")",
        "df[\"intent\"].value_counts()",
    ),
    code(
        "# Peek at one sample's raw audio (a sine wave) and video (a small RGB frame).",
        "sample = df.iloc[0]",
        "clip = np.load(PROJECT_ROOT / sample[\"audio_path\"])",
        "frame = np.load(PROJECT_ROOT / sample[\"video_path\"])",
        "print(f\"Sample: {sample['text']!r} -> intent={sample['intent']}\")",
        "print(f\"Audio clip shape: {clip.shape}, dtype: {clip.dtype}\")",
        "print(f\"Video frame shape: {frame.shape}, dtype: {frame.dtype}\")",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3))",
        "axes[0].plot(clip[:200])",
        "axes[0].set_title(\"First 200 samples of the fake audio wave\")",
        "axes[1].imshow(frame)",
        "axes[1].set_title(f\"Fake video frame (intent={sample['intent']})\")",
        "axes[1].axis(\"off\")",
        "fig.tight_layout()",
        "plt.show()",
    ),
    md(
        "## Step 2 - Split into train / validation / test",
        "",
        "Identical split logic (same `test_size`, `random_state`, `stratify`) to Milestones 0 and 1, so results",
        "stay comparable across notebooks.",
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
        "## Step 3 - Extract features from each modality and fuse them",
        "",
        "- **Text**: same `TfidfVectorizer` as Milestone 0.",
        "- **Audio**: FFT (Fast Fourier Transform) recovers which pitch is loudest in each fake clip --",
        "  its *dominant frequency* -- plus its RMS *energy* (how loud it is overall). 2 numbers per sample.",
        "- **Video**: mean red/green/blue plus how much the brightness varies across the frame. 4 numbers per sample.",
        "",
        "**Concatenation fusion** just means gluing these three feature vectors end to end into one longer vector",
        "per sample, before the classifier ever sees them.",
    ),
    code(
        "def extract_audio_features(clip):",
        "    fft_vals = np.fft.rfft(clip)",
        "    fft_freqs = np.fft.rfftfreq(len(clip), d=1 / SAMPLE_RATE)",
        "    dominant_freq = fft_freqs[np.argmax(np.abs(fft_vals))]",
        "    energy = np.sqrt(np.mean(clip ** 2))",
        "    return np.array([dominant_freq, energy])",
        "",
        "",
        "def extract_video_features(frame):",
        "    frame = frame.astype(np.float32)",
        "    mean_r, mean_g, mean_b = frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()",
        "    brightness_variance = frame.mean(axis=2).var()",
        "    return np.array([mean_r, mean_g, mean_b, brightness_variance])",
        "",
        "",
        "def build_audio_features(frame_df):",
        "    return np.array([extract_audio_features(np.load(PROJECT_ROOT / p)) for p in frame_df[\"audio_path\"]])",
        "",
        "",
        "def build_video_features(frame_df):",
        "    return np.array([extract_video_features(np.load(PROJECT_ROOT / p)) for p in frame_df[\"video_path\"]])",
    ),
    code(
        "# See the raw feature scales BEFORE scaling -- this is why StandardScaler matters.",
        "sample_audio = build_audio_features(train.iloc[:5])",
        "sample_video = build_video_features(train.iloc[:5])",
        "print(\"Raw audio features (dominant_freq, energy), first 5 rows:\")",
        "print(sample_audio)",
        "print(\"\\nRaw video features (R, G, B, brightness_var), first 5 rows:\")",
        "print(sample_video)",
        "print(\"\\nNotice: audio frequency is in the hundreds, RGB is 0-255, but TF-IDF values are 0-1ish.\")",
        "print(\"Fed in raw, the biggest numbers would unfairly dominate what the classifier learns --\")",
        "print(\"that's exactly what StandardScaler below fixes.\")",
    ),
    code(
        "tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)",
        "text_feats_train = tfidf.fit_transform(train[\"text\"]).toarray()",
        "audio_feats_train = build_audio_features(train)",
        "video_feats_train = build_video_features(train)",
        "",
        "fused_train_raw = np.hstack([text_feats_train, audio_feats_train, video_feats_train])",
        "print(f\"Fused feature vector size: {fused_train_raw.shape[1]} \"",
        "      f\"(text={text_feats_train.shape[1]}, audio=2, video=4)\")",
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
    ),
    code(
        "def transform_features(frame_df):",
        "    text_feats = tfidf.transform(frame_df[\"text\"]).toarray()",
        "    audio_feats = build_audio_features(frame_df)",
        "    video_feats = build_video_features(frame_df)",
        "    fused = np.hstack([text_feats, audio_feats, video_feats])",
        "    return scaler.transform(fused)",
        "",
        "",
        "fused_test = transform_features(test)",
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
        "fig, ax = plt.subplots(figsize=(6, 5))",
        "im = ax.imshow(cm, cmap=\"Blues\")",
        "ax.set_xticks(range(len(labels)), labels, rotation=45, ha=\"right\")",
        "ax.set_yticks(range(len(labels)), labels)",
        "ax.set_xlabel(\"predicted\")",
        "ax.set_ylabel(\"true\")",
        "ax.set_title(\"Milestone 2 (toy fusion) confusion matrix\")",
        "for i in range(len(labels)):",
        "    for j in range(len(labels)):",
        "        ax.text(j, i, str(cm[i, j]), ha=\"center\", va=\"center\",",
        "                color=\"white\" if cm[i, j] > cm.max() / 2 else \"black\")",
        "fig.colorbar(im, ax=ax, shrink=0.8, label=\"count\")",
        "fig.tight_layout()",
        "plt.show()",
    ),
    md(
        "### Compare against the Milestone 0 text-only baseline",
    ),
    code(
        "comparison = pd.DataFrame([",
        "    {\"model\": \"Text-only (M0)\", \"accuracy\": 0.550, \"macro_f1\": 0.521},",
        "    {\"model\": \"Text+audio+video fusion (M2, toy data)\", \"accuracy\": acc, \"macro_f1\": macro_f1},",
        "]).set_index(\"model\")",
        "comparison",
    ),
    md(
        "**Interpretation:** because the fake audio and video signals were deliberately tied to the intent",
        "labels, fusion should meet or beat the text-only baseline here. If it doesn't, that's a signal something",
        "is off in the feature extraction or scaling -- worth debugging before trusting this pattern on real data.",
        "Remember: this result says nothing yet about how well fusion will work on REAL audio/video, where the",
        "signal is far messier than a clean sine wave -- it only proves the fusion *mechanics* are correct.",
    ),
    md(
        "## Look at the mistakes",
    ),
    code(
        "results_df = pd.DataFrame({\"text\": test[\"text\"], \"true\": truth, \"predicted\": preds})",
        "",
        "RESULTS_DIR.mkdir(exist_ok=True)",
        "out_path = RESULTS_DIR / \"toy_multimodal_predictions.csv\"",
        "results_df.to_csv(out_path, index=False)",
        "print(f\"Saved every test prediction to {out_path.relative_to(PROJECT_ROOT)}\")",
        "",
        "mistakes = results_df[results_df[\"true\"] != results_df[\"predicted\"]]",
        "print(f\"\\n{len(mistakes)} of {len(results_df)} test sentences were misclassified:\")",
        "mistakes",
    ),
    md(
        "## Exercises",
        "",
        "1. **Predict before running:** comment out the `scaler = StandardScaler()` / `fused_train = scaler...`",
        "   lines in Step 3 (use `fused_train = fused_train_raw` instead, and update Step 4's",
        "   `transform_features` to skip `scaler.transform` too). Do you expect accuracy to go up, down, or stay",
        "   about the same without scaling? Run it and check.",
        "2. In `src/generate_toy_multimodal.py`, `BASE_FREQUENCIES` spaces each intent 100Hz apart. Try making",
        "   them all much closer together (e.g. 10Hz apart) by editing that dict, rerun the generator and this",
        "   notebook, and see how much harder that makes the audio signal to use.",
    ),
    md(
        "## Pitfalls and extensions",
        "",
        "**Common mistake:** concatenating features with wildly different scales (frequency in the hundreds vs.",
        "TF-IDF values near 0-1) without scaling first. The model can end up dominated by whichever modality",
        "happens to produce the largest raw numbers -- not whichever modality is actually most informative.",
        "`StandardScaler` (fit on train, applied to val/test) fixes this by putting every feature on the same",
        "footing before the classifier sees it.",
        "",
        "**Extension (next increment):** this notebook's 'encoders' are hand-written classic-signal functions,",
        "not learned models -- deliberately, to avoid any downloads here. The real next step is swapping",
        "`extract_audio_features`/`extract_video_features` for real frozen pretrained encoders (e.g. wav2vec2 for",
        "audio, a pretrained CNN for video) once real MIntRec audio/video data is acquired -- a separate,",
        "larger increment that needs its own download-size confirmation.",
    ),
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells to {NB_PATH}")
```

- [ ] **Step 3: Run the builder script**

```bash
python <path-to-scratchpad>/build_m2_notebook.py
```

Expected: `Wrote 20 cells to ...notebooks\milestone2_toy_fusion.ipynb`.

- [ ] **Step 4: Execute the notebook top-to-bottom**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi jupyter nbconvert --to notebook --execute --inplace "notebooks/milestone2_toy_fusion.ipynb"
```

Expected: `[NbConvertApp] Writing ... bytes to ...notebooks\milestone2_toy_fusion.ipynb` with no Python traceback in the command output (warnings about symlinks/TCP kernels, like in M0/M1, are expected and harmless).

- [ ] **Step 5: Verify no errors in the executed notebook**

```bash
python -c "
import json
nb = json.load(open('notebooks/milestone2_toy_fusion.ipynb', encoding='utf-8'))
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

Use the `commit` skill (it will scan the notebook's embedded outputs for secrets too, same as the M0/M1 notebooks). Stage `notebooks/milestone2_toy_fusion.ipynb` only (the builder script stays in the scratchpad, not part of the repo, matching how the M0/M1 notebooks were built). Suggested message subject: `feat: add Milestone 2 toy fusion notebook`.

---

## Final check

After Task 3, confirm the three deliverables together tell a complete story: `src/generate_toy_multimodal.py` (data), `src/train_multimodal_toy.py` (terminal-runnable pipeline), `notebooks/milestone2_toy_fusion.ipynb` (narrated version with plots). Consider updating the M2 checklist entry in the project's `CLAUDE.md` to note this increment is done and that real audio/video + real pretrained encoders are still pending — but confirm with the user first, the same way M1's checkbox was handled.
