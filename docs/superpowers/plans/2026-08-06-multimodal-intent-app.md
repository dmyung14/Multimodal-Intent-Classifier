# Multimodal Intent Inference App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack app (FastAPI backend + Next.js frontend) that serves this project's already-trained M3/M5 models, accepting text/video input and returning a predicted intent plus an honest, evidence-based explanation.

**Architecture:** Two local processes. A one-time export script trains and saves checkpoints for the models this app serves (none of M0-M5's scripts persist weights today). A FastAPI backend loads those checkpoints once at startup, extracts audio/video features live from uploaded clips using M2's exact frozen-encoder pipeline, and serves `POST /predict`. A Next.js frontend renders the form and results. Built in two phases: Phase 1 is a minimum working vertical slice (2 models, minimal UI); Phase 2 widens to all 8 models and the full designed UI.

**Tech Stack:** Python (FastAPI, uvicorn, joblib, torch, scikit-learn — all inside the existing `mmi` conda env) for the backend; Next.js/React (npm) for the frontend.

## Global Constraints

- **Workspace: directly on the project directory, no git worktree** — matches every prior milestone plan in this project (M2-M5), user has already consented to this pattern in this session.
- **Autonomous authorization:** user said "move onto the implementation plan, automate the entire process, and don't prompt me until it is finished." No per-task confirmation gates. Document decisions with reasoning in commit messages/task reports instead of asking.
- **Determinism:** `modality_ablation.py` uses `SEED = 0` (`LogisticRegression(random_state=SEED)`); `train_misa.py` uses `SEED = 0` (`torch.manual_seed`) and `random_state=42` (both `train_test_split` calls). The export script (Tasks 1 and 7) must call those scripts' existing functions **exactly as they exist** — never reimplement splitting/training logic — and must verify the re-exported checkpoint reproduces the already-recorded real numbers below, failing loudly (not silently) if it doesn't.
- **Real, already-recorded numbers every export must reproduce** (source: `results/modality_ablation.csv` and `CLAUDE.md`; tolerance ±0.005 on accuracy and macro-F1 each):
  | combo | accuracy | macro_f1 |
  |---|---|---|
  | T | 0.4206 | 0.2941 |
  | A | 0.1495 | 0.0687 |
  | V | 0.1121 | 0.0361 |
  | TA | 0.4112 | 0.2466 |
  | TV | 0.3364 | 0.2537 |
  | AV | 0.1963 | 0.0959 |
  | TAV | 0.3458 | 0.2083 |
  | MISA | 0.333 | 0.295 |
- **No M4 anywhere in this plan.** M4's cross-modal model is explicitly excluded (dialogue-context dependency a single-shot input doesn't have) per the design spec.
- **No speech-to-text anywhere.** Text is always user-typed; state this plainly in UI copy.
- **Fail loudly with a fix-command pointer** on any missing prerequisite (missing checkpoint, missing input) — same convention as every M0-M5 script's `SystemExit`.
- **New dependencies this plan installs** (already verified available/needed on this machine — no confirmation gate per the autonomous authorization, but document what's added): `fastapi`, `uvicorn`, `python-multipart`, `pytest` via `pip install` inside the `mmi` conda env (`joblib` and `httpx` already present). `next`, `react`, `react-dom` via `npm install` inside `frontend/` (Node v24.16.0 / npm 11.13.0 already installed on this machine — confirmed, no Node install needed).
- **ffmpeg/ffprobe** already available inside the `mmi` conda env (confirmed: ffmpeg 8.1.2) — invoke via `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi ffmpeg ...` or from within a script already running in that env.
- **Test fixture clip:** `data/dialogue/raw_clips/S04/E04/374.mp4` (320KB, already downloaded) — use this exact file for any backend test needing a real video upload.
- Invoke the conda env via `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python <script>` (Git Bash, POSIX paths) — never invoke the env's `python.exe` directly (crashes with `STATUS_STACK_BUFFER_OVERRUN`), never use `conda run -n mmi python -c "<multiline>"` (broken — write a temp `.py` file instead).

---

## Phase 1: minimum working vertical slice (2 models, minimal frontend)

### Task 1: `scripts/export_checkpoints.py` — export T and MISA checkpoints

**Files:**
- Create: `scripts/export_checkpoints.py`
- Create: `scripts/__init__.py` (empty, so this can be imported as a package later if needed)

**Interfaces:**
- Consumes: `src/modality_ablation.py`'s `load_data()`, `split_data(df)`, `build_features(combo, train_val, test, tfidf)`, `SEED` (imported, not reimplemented). `src/train_misa.py`'s `load_data()`, `split_data(df)`, `featurize(train, val, test)`, `to_tensors(...)`, `train_misa(train_tensors, val_tensors)`, `evaluate(model, test_tensors, id2label, test_df)`, `HIDDEN_DIM` (imported, not reimplemented). `src/misa_model.py`'s `MISAModel`.
- Produces: `models/lr_T.joblib` — a `joblib`-dumped dict `{"combo": "T", "tfidf": TfidfVectorizer|None, "scaler": StandardScaler, "clf": LogisticRegression}`. `models/misa.pt` — a `torch.save`-dumped dict `{"state_dict": ..., "tfidf": TfidfVectorizer, "id2label": dict[int,str], "text_dim": int, "hidden_dim": int, "num_classes": int}`. Later tasks (3, 8) load these exact shapes.

- [ ] **Step 1: Write `scripts/export_checkpoints.py`**

```python
"""
scripts/export_checkpoints.py

One-time export step: trains M5's "T" combo and M3's MISA using their
EXISTING, already-reviewed training code (no reimplementation), then
saves the fitted objects the backend needs to serve them without
retraining on every app startup.

Includes a determinism check: re-running each exported checkpoint's own
scoring must reproduce the already-recorded real numbers from
CLAUDE.md/results/modality_ablation.csv. If it doesn't match, this
script fails loudly -- a silently different model than the one this app
claims to serve is worse than no model at all.

HOW TO RUN
    conda activate mmi
    python scripts/export_checkpoints.py

    Phase 1: exports 2 checkpoints (T, MISA) to models/. Fast (T: a few
    seconds; MISA: under a minute, same as running src/train_misa.py
    directly).
"""
import sys
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import modality_ablation as ma  # noqa: E402
import train_misa as tm  # noqa: E402
from misa_model import MISAModel  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

# Real, already-recorded numbers this export must reproduce (see this
# plan's Global Constraints table -- source: results/modality_ablation.csv
# and CLAUDE.md).
EXPECTED = {
    "T": {"accuracy": 0.4206, "macro_f1": 0.2941},
    "MISA": {"accuracy": 0.333, "macro_f1": 0.295},
}
TOLERANCE = 0.005


def _verify(name, acc, macro_f1):
    expected = EXPECTED[name]
    acc_ok = abs(acc - expected["accuracy"]) <= TOLERANCE
    f1_ok = abs(macro_f1 - expected["macro_f1"]) <= TOLERANCE
    if not (acc_ok and f1_ok):
        raise SystemExit(
            f"\nDETERMINISM CHECK FAILED for '{name}':\n"
            f"  expected accuracy={expected['accuracy']:.4f} macro_f1={expected['macro_f1']:.4f}\n"
            f"  got      accuracy={acc:.4f} macro_f1={macro_f1:.4f}\n"
            "This means the export re-ran training differently than the "
            "original script did (different seed, different data, or a "
            "different code path). Do not serve this checkpoint until this "
            "is understood and fixed.\n"
        )
    print(f"Determinism check passed for '{name}': matches recorded real results.")


def export_lr_combo(combo):
    print("=" * 70)
    print(f"EXPORTING M5 COMBO '{combo}'")
    print("=" * 70)
    df = ma.load_data()
    train_val, test = ma.split_data(df)
    all_labels = sorted(df["target_intent"].unique())

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train, x_test = ma.build_features(combo, train_val, test, tfidf)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=1000, random_state=ma.SEED)
    clf.fit(x_train_scaled, train_val["target_intent"])

    preds = clf.predict(x_test_scaled)
    truth = test["target_intent"].tolist()
    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro", labels=all_labels, zero_division=0)
    _verify(combo, acc, macro_f1)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"lr_{combo}.joblib"
    saved_tfidf = tfidf if "T" in combo else None
    joblib.dump({"combo": combo, "tfidf": saved_tfidf, "scaler": scaler, "clf": clf}, out_path)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)} (accuracy={acc:.4f}, macro_f1={macro_f1:.4f})\n")


def export_misa():
    print("=" * 70)
    print("EXPORTING M3 MISA")
    print("=" * 70)
    df = tm.load_data()
    train, val, test = tm.split_data(df)
    (text_train, audio_train, video_train), (text_val, audio_val, video_val), \
        (text_test, audio_test, video_test), tfidf = tm.featurize(train, val, test)

    labels_sorted = sorted(df["intent"].unique())
    label2id = {name: i for i, name in enumerate(labels_sorted)}
    id2label = {i: name for name, i in label2id.items()}

    train_tensors = tm.to_tensors(text_train, audio_train, video_train, train["intent"].tolist(), label2id)
    val_tensors = tm.to_tensors(text_val, audio_val, video_val, val["intent"].tolist(), label2id)
    test_tensors = tm.to_tensors(text_test, audio_test, video_test, test["intent"].tolist(), label2id)

    model = tm.train_misa(train_tensors, val_tensors)
    acc, macro_f1 = tm.evaluate(model, test_tensors, id2label, test)
    _verify("MISA", acc, macro_f1)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "misa.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "tfidf": tfidf,
            "id2label": id2label,
            "text_dim": text_train.shape[1],
            "hidden_dim": tm.HIDDEN_DIM,
            "num_classes": len(labels_sorted),
        },
        out_path,
    )
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)} (accuracy={acc:.4f}, macro_f1={macro_f1:.4f})\n")


def main():
    export_lr_combo("T")
    export_misa()
    print("=" * 70)
    print("DONE. Both checkpoints verified and saved to models/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/__init__.py`**

Empty file.

- [ ] **Step 3: Run it**

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python scripts/export_checkpoints.py`

Expected: both determinism checks pass (printed "Determinism check passed for 'T'" and "...for 'MISA'"), `models/lr_T.joblib` and `models/misa.pt` both exist. If a determinism check fails, do not proceed — investigate whether `modality_ablation.py`/`train_misa.py` were called with any deviation from their original seeded calls before touching anything else.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_checkpoints.py scripts/__init__.py
git commit -m "feat: add checkpoint export script for T and MISA models"
```

(`models/` is git-ignored — do not add checkpoint files to git.)

---

### Task 2: `backend/extraction.py` — live audio/video feature extraction

**Files:**
- Create: `backend/__init__.py` (empty)
- Create: `backend/extraction.py`

**Interfaces:**
- Consumes: nothing from other tasks in this plan (mirrors `src/generate_mintrec_multimodal.py`'s ffmpeg extraction and `src/extract_mintrec_embeddings.py`'s encoder logic, parameterized for a single arbitrary uploaded file rather than a dataset row).
- Produces: `load_encoders() -> dict` (call once, at backend startup). `extract_live_features(video_path: Path, encoders: dict, work_dir: Path) -> {"audio": np.ndarray shape (768,), "video": np.ndarray shape (512,)}`. Task 3 and Task 5 both call these.

- [ ] **Step 1: Write `backend/extraction.py`**

```python
"""
backend/extraction.py

Live audio/video feature extraction for the app's /predict endpoint.
Mirrors two already-reviewed Milestone 2 scripts exactly, just
parameterized for an arbitrary uploaded file instead of a fixed dataset
row:
  - src/generate_mintrec_multimodal.py's extract_audio/extract_frames
    (ffmpeg invocations: mono 16kHz WAV; 5 evenly-spaced JPEG frames)
  - src/extract_mintrec_embeddings.py's embed_audio/embed_video
    (frozen wav2vec2-base / ResNet18, mean-pooled)

Both encoders are FROZEN here too -- .eval(), no gradients, matching
this project's "prefer frozen encoders" rule.
"""
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

AUDIO_SAMPLE_RATE = 16000
N_FRAMES = 5
AUDIO_CKPT = "facebook/wav2vec2-base"


def load_encoders():
    """Load both frozen encoders once. Call at FastAPI startup, not per request."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    audio_model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    audio_model.eval()

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    video_model = torchvision.models.resnet18(weights=weights)
    video_model.fc = torch.nn.Identity()
    video_model.eval()
    video_transforms = weights.transforms()

    return {
        "audio_feature_extractor": feature_extractor,
        "audio_model": audio_model,
        "video_model": video_model,
        "video_transforms": video_transforms,
    }


def _extract_audio_wav(clip_path, work_dir):
    out_path = Path(work_dir) / "audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-vn",
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path


def _get_duration_seconds(clip_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(clip_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_frames(clip_path, work_dir):
    frame_dir = Path(work_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = _get_duration_seconds(clip_path)
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


def _read_wav_as_array(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _embed_audio(wav_path, feature_extractor, model):
    audio = _read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.numpy()


def _embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.mean(embeddings, axis=0)


def extract_live_features(video_path, encoders, work_dir):
    """
    video_path: Path to an uploaded mp4/webm file.
    encoders: dict returned by load_encoders().
    work_dir: Path to a scratch directory for this request's intermediate
              WAV/JPEG files (caller creates and cleans it up).
    Returns: {"audio": np.ndarray(768,), "video": np.ndarray(512,)}
    """
    wav_path = _extract_audio_wav(video_path, work_dir)
    frame_dir = _extract_frames(video_path, work_dir)
    audio_embedding = _embed_audio(
        wav_path, encoders["audio_feature_extractor"], encoders["audio_model"]
    )
    video_embedding = _embed_video(
        frame_dir, encoders["video_model"], encoders["video_transforms"]
    )
    return {"audio": audio_embedding, "video": video_embedding}
```

- [ ] **Step 2: Create `backend/__init__.py`**

Empty file.

- [ ] **Step 3: Verify shape/magnitude consistency against a real cached embedding**

Write a throwaway verification script (do not commit it) that runs `extract_live_features` on the fixture clip (`data/dialogue/raw_clips/S04/E04/374.mp4`) and compares its output shape and rough magnitude against a real cached embedding from `data/mintrec_multimodal/embeddings/` (any `.npz` file — e.g. `data/mintrec_multimodal/embeddings/S04_E01_181.npz`, loaded via `np.load(...)["audio"]`/`["video"]`). Confirm: `audio` embedding shape is `(768,)` in both, `video` embedding shape is `(512,)` in both, and neither is all-zeros or contains NaN. This is the design spec's explicitly-flagged "MISA embedding-dimension consistency" risk — verify it directly rather than assuming the live path matches the cached path.

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python <path-to-throwaway-script>.py`

Expected: printed shapes `(768,)` and `(512,)` for both the live-extracted and cached embeddings, no NaN, no assertion failure.

- [ ] **Step 4: Commit**

```bash
git add backend/__init__.py backend/extraction.py
git commit -m "feat: add live audio/video feature extraction for the backend"
```

---

### Task 3: `backend/registry.py` + `backend/inference.py` — load checkpoints, run predictions

**Files:**
- Create: `backend/registry.py`
- Create: `backend/inference.py`

**Interfaces:**
- Consumes: `models/lr_T.joblib` and `models/misa.pt` (Task 1's output, exact shapes documented in Task 1's Interfaces block). `src/misa_model.py`'s `MISAModel`.
- Produces: `load_registry() -> dict` (keys: any of `"T"`/`"A"`/`"V"`/`"TA"`/`"TV"`/`"AV"`/`"TAV"`/`"MISA"` that have been exported and loaded; Phase 1 only has `"T"` and `"MISA"`). `MODEL_REQUIREMENTS: dict[str, {"needs_text": bool, "needs_video": bool}]` for all 8 possible IDs (used by Task 5's endpoint for input validation). `predict(model_id: str, text: str|None, live_features: dict|None, registry: dict) -> {"predicted_intent": str, "confidence": float, "probabilities": dict[str, float]}` (Task 5 calls this).

- [ ] **Step 1: Write `backend/registry.py`**

```python
"""
backend/registry.py

Loads every trained model checkpoint this app can serve, once, at
startup. Fails loudly with an exact fix-command pointer if any expected
checkpoint is missing -- same convention every M0-M5 script uses for
missing inputs.
"""
import sys
from pathlib import Path

import joblib
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misa_model import MISAModel  # noqa: E402

# Phase 1: only "T" is exported/served. Phase 2 (Task 8) extends this to
# all 7 of M5's modality combinations -- everything else about this file
# stays the same, this list is the only thing that grows.
LR_COMBOS = ["T"]

MODEL_REQUIREMENTS = {
    "T": {"needs_text": True, "needs_video": False},
    "A": {"needs_text": False, "needs_video": True},
    "V": {"needs_text": False, "needs_video": True},
    "TA": {"needs_text": True, "needs_video": True},
    "TV": {"needs_text": True, "needs_video": True},
    "AV": {"needs_text": False, "needs_video": True},
    "TAV": {"needs_text": True, "needs_video": True},
    "MISA": {"needs_text": True, "needs_video": True},
}


def load_registry():
    registry = {}
    missing = []

    for combo in LR_COMBOS:
        path = MODELS_DIR / f"lr_{combo}.joblib"
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        registry[combo] = joblib.load(path)

    misa_path = MODELS_DIR / "misa.pt"
    if not misa_path.exists():
        missing.append(str(misa_path.relative_to(PROJECT_ROOT)))
    else:
        bundle = torch.load(misa_path, weights_only=False)
        model = MISAModel(
            text_dim=bundle["text_dim"],
            hidden_dim=bundle["hidden_dim"],
            num_classes=bundle["num_classes"],
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        registry["MISA"] = {"model": model, "tfidf": bundle["tfidf"], "id2label": bundle["id2label"]}

    if missing:
        raise RuntimeError(
            "\nMissing model checkpoint(s):\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\n\nRun the export script first:\n    python scripts/export_checkpoints.py\n"
        )

    return registry
```

- [ ] **Step 2: Write `backend/inference.py`**

```python
"""
backend/inference.py

Runs a single prediction through whichever model the request asked for.
Two model families, two different forward passes, one shared return shape.
"""
import numpy as np
import torch


def predict(model_id, text, live_features, registry):
    """
    model_id: a key present in `registry` (e.g. "T", "MISA").
    text: str or None.
    live_features: {"audio": np.ndarray(768,), "video": np.ndarray(512,)} or None.
    registry: as returned by backend.registry.load_registry().
    Returns: {"predicted_intent": str, "confidence": float, "probabilities": dict[str, float]}
    """
    if model_id == "MISA":
        return _predict_misa(text, live_features, registry["MISA"])
    return _predict_lr(model_id, text, live_features, registry[model_id])


def _predict_lr(combo, text, live_features, bundle):
    parts = []
    if "T" in combo:
        parts.append(bundle["tfidf"].transform([text]).toarray())
    if "A" in combo:
        parts.append(live_features["audio"].reshape(1, -1))
    if "V" in combo:
        parts.append(live_features["video"].reshape(1, -1))
    x = np.hstack(parts)
    x_scaled = bundle["scaler"].transform(x)

    clf = bundle["clf"]
    probs = clf.predict_proba(x_scaled)[0]
    classes = clf.classes_
    top_idx = int(np.argmax(probs))
    return {
        "predicted_intent": str(classes[top_idx]),
        "confidence": float(probs[top_idx]),
        "probabilities": {str(cls): float(p) for cls, p in zip(classes, probs)},
    }


def _predict_misa(text, live_features, misa_entry):
    model = misa_entry["model"]
    tfidf = misa_entry["tfidf"]
    id2label = misa_entry["id2label"]

    text_feat = torch.from_numpy(tfidf.transform([text]).toarray().astype(np.float32))
    audio_feat = torch.from_numpy(live_features["audio"].astype(np.float32)).unsqueeze(0)
    video_feat = torch.from_numpy(live_features["video"].astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(text_feat, audio_feat, video_feat)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    top_idx = int(np.argmax(probs))
    return {
        "predicted_intent": id2label[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {id2label[i]: float(p) for i, p in enumerate(probs)},
    }
```

- [ ] **Step 3: Smoke test both paths**

Write a throwaway script (do not commit) that: calls `load_registry()`, then calls `predict("T", "I guess we should get going now", None, registry)` and prints the result, then (reusing Task 2's `load_encoders`/`extract_live_features` on the fixture clip) calls `predict("MISA", "I guess we should get going now", live_features, registry)` and prints that result too.

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python <path-to-throwaway-script>.py`

Expected: both calls print a dict with a real `predicted_intent` string (one of MIntRec's 20 intent labels), a `confidence` between 0 and 1, and a `probabilities` dict with entries summing to ~1.0. No exceptions.

- [ ] **Step 4: Commit**

```bash
git add backend/registry.py backend/inference.py
git commit -m "feat: add model registry and prediction dispatch"
```

---

### Task 4: `backend/explanation.py` — calibration caveat + word attribution

**Files:**
- Create: `backend/explanation.py`

**Interfaces:**
- Consumes: `bundle` (a registry entry, as produced by Task 3's `load_registry()`), `prediction` (Task 3's `predict()` return shape).
- Produces: `build_explanation(model_id: str, text: str|None, bundle: dict, prediction: dict) -> {"calibration_caveat": str, "top_words": list[{"word": str, "weight": float}]|None}`. Task 5 calls this.

- [ ] **Step 1: Write `backend/explanation.py`**

```python
"""
backend/explanation.py

Every prediction gets two honest, evidence-based explanation layers
(per docs/superpowers/specs/2026-08-06-multimodal-intent-app-design.md):

1. A calibration caveat -- a STATIC fact about the model family, not
   recomputed per request (calibration is a property of the model, not
   of any single prediction). Only "T" has an actual measured ECE
   (Milestone 5); every other model/combo is honestly labeled as
   "not measured" rather than reusing T's specific number for a model
   that was never calibration-tested.
2. Top contributing words -- ONLY for M5's text-involving
   LogisticRegression combos (T, TA, TV, TAV). Computed as each present
   word's real contribution to the predicted class's linear score:
   coefficient x SCALED feature value (the StandardScaler's transform is
   applied before the score is computed during training, so leaving it
   out here would misrepresent the model's actual math). None for MISA
   (a neural net -- no such direct, honest attribution without adding a
   new technique like SHAP, which is out of scope) and for non-text
   combos (A, V, AV -- no text to attribute to).
"""
import numpy as np

CALIBRATION_CAVEATS = {
    "T": (
        "This model was measured (Milestone 5) to be systematically "
        "overconfident: its average stated confidence (0.485) is higher "
        "than its actual accuracy (0.421) on held-out test data "
        "(Expected Calibration Error = 0.089). Treat the confidence "
        "number above as an upper bound, not a precise probability."
    ),
}
DEFAULT_CAVEAT = (
    "This model's calibration (whether its confidence numbers match its "
    "real accuracy) has not been measured. Treat the confidence number "
    "above as a rough signal only, not a calibrated probability."
)


def top_contributing_words(model_id, text, bundle, prediction):
    if model_id == "MISA" or "T" not in model_id:
        return None

    tfidf = bundle["tfidf"]
    scaler = bundle["scaler"]
    clf = bundle["clf"]

    vocab = tfidf.get_feature_names_out()
    n_text_features = len(vocab)

    class_idx = list(clf.classes_).index(prediction["predicted_intent"])
    coefs = clf.coef_[class_idx][:n_text_features]
    means = scaler.mean_[:n_text_features]
    scales = scaler.scale_[:n_text_features]

    tfidf_vec = tfidf.transform([text]).toarray()[0]
    scaled_vec = (tfidf_vec - means) / scales

    present_idx = np.nonzero(tfidf_vec)[0]
    contributions = [
        {"word": vocab[i], "weight": float(coefs[i] * scaled_vec[i])}
        for i in present_idx
    ]
    contributions.sort(key=lambda c: c["weight"], reverse=True)
    return contributions[:10]


def build_explanation(model_id, text, bundle, prediction):
    return {
        "calibration_caveat": CALIBRATION_CAVEATS.get(model_id, DEFAULT_CAVEAT),
        "top_words": top_contributing_words(model_id, text, bundle, prediction),
    }
```

- [ ] **Step 2: Smoke test**

Write a throwaway script (do not commit) that loads the registry (Task 3), runs `predict("T", "I guess we should get going now", None, registry)`, then calls `build_explanation("T", "I guess we should get going now", registry["T"], <that prediction>)` and prints the result. Also call it with `model_id="MISA"` (using a MISA prediction from Task 3's smoke test) and confirm `top_words` is `None`.

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python <path-to-throwaway-script>.py`

Expected: for `"T"`, a non-empty `top_words` list of `{"word", "weight"}` dicts, sorted descending by weight, and a `calibration_caveat` mentioning ECE 0.089. For `"MISA"`, `top_words` is `None` and `calibration_caveat` is the generic "not measured" text.

- [ ] **Step 3: Commit**

```bash
git add backend/explanation.py
git commit -m "feat: add calibration caveat and word-attribution explanations"
```

---

### Task 5: `backend/app.py` + tests — FastAPI app

**Files:**
- Create: `backend/app.py`
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/tests/test_predict.py`

**Interfaces:**
- Consumes: Task 2's `load_encoders`/`extract_live_features`, Task 3's `load_registry`/`predict`/`MODEL_REQUIREMENTS`, Task 4's `build_explanation`.
- Produces: `POST /predict` HTTP endpoint (multipart form: `model_choice: str` required, `text: str` optional, `video: UploadFile` optional). JSON response shape: `{"model_choice": str, "predicted_intent": str, "confidence": float, "probabilities": dict[str, float], "explanation": {"calibration_caveat": str, "top_words": list|None}}`. Task 6 and Task 9 (frontend) call this over HTTP.

- [ ] **Step 1: Install new backend dependencies**

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi pip install fastapi uvicorn python-multipart pytest`

(`joblib` and `httpx` are already present in the `mmi` env — confirmed before writing this plan.)

- [ ] **Step 2: Write `backend/app.py`**

```python
"""
backend/app.py

FastAPI app serving this project's trained M3/M5 models.

HOW TO RUN
    conda activate mmi
    python scripts/export_checkpoints.py   # once, if models/ is empty
    uvicorn backend.app:app --reload --port 8000
"""
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.registry import load_registry, MODEL_REQUIREMENTS
from backend.extraction import load_encoders, extract_live_features
from backend.inference import predict
from backend.explanation import build_explanation

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["registry"] = load_registry()
    STATE["encoders"] = load_encoders()
    yield


app = FastAPI(title="Multimodal Intent Inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_endpoint(
    model_choice: str = Form(...),
    text: Optional[str] = Form(None),
    video: Optional[UploadFile] = File(None),
):
    if model_choice not in STATE["registry"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown or unavailable model_choice '{model_choice}'. "
                f"Available: {sorted(STATE['registry'])}"
            },
        )

    requirements = MODEL_REQUIREMENTS[model_choice]
    if requirements["needs_text"] and not text:
        return JSONResponse(
            status_code=400,
            content={"error": f"model_choice '{model_choice}' requires text input."},
        )
    if requirements["needs_video"] and video is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"model_choice '{model_choice}' requires a video file upload."},
        )

    live_features = None
    if requirements["needs_video"]:
        work_dir = Path(tempfile.mkdtemp(prefix="mmi_predict_"))
        try:
            video_path = work_dir / "upload.mp4"
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            try:
                live_features = extract_live_features(video_path, STATE["encoders"], work_dir)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to process uploaded video: {exc}"},
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    bundle = STATE["registry"][model_choice]
    prediction = predict(model_choice, text, live_features, STATE["registry"])
    explanation = build_explanation(model_choice, text, bundle, prediction)

    return {
        "model_choice": model_choice,
        "predicted_intent": prediction["predicted_intent"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "explanation": explanation,
    }
```

- [ ] **Step 3: Create `backend/tests/__init__.py`**

Empty file.

- [ ] **Step 4: Write `backend/tests/test_predict.py`**

```python
"""
backend/tests/test_predict.py

Integration tests for POST /predict. Prerequisite: checkpoints must
already be exported (python scripts/export_checkpoints.py) -- these
tests load the real app, including its real startup (real encoders,
real checkpoints), not mocks.

HOW TO RUN
    conda activate mmi
    python scripts/export_checkpoints.py   # once, if not already done
    pytest backend/tests/test_predict.py -v
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_CLIP = PROJECT_ROOT / "data" / "dialogue" / "raw_clips" / "S04" / "E04" / "374.mp4"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_missing_model_choice_returns_422(client):
    response = client.post("/predict", data={"text": "hello"})
    assert response.status_code == 422  # FastAPI's own required-field validation


def test_unknown_model_choice_returns_400(client):
    response = client.post("/predict", data={"model_choice": "NOT_A_REAL_MODEL", "text": "hello"})
    assert response.status_code == 400
    assert "Unknown or unavailable" in response.json()["error"]


def test_t_combo_missing_text_returns_400(client):
    response = client.post("/predict", data={"model_choice": "T"})
    assert response.status_code == 400
    assert "requires text" in response.json()["error"]


def test_misa_missing_video_returns_400(client):
    response = client.post(
        "/predict", data={"model_choice": "MISA", "text": "I guess we should get going"}
    )
    assert response.status_code == 400
    assert "requires a video" in response.json()["error"]


def test_t_combo_happy_path(client):
    response = client.post(
        "/predict",
        data={"model_choice": "T", "text": "I guess we should get going now"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "T"
    assert isinstance(body["predicted_intent"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]["calibration_caveat"]
    assert body["explanation"]["top_words"] is not None


def test_misa_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "MISA", "text": "I guess we should get going now"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "MISA"
    assert isinstance(body["predicted_intent"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]["top_words"] is None  # MISA has no word attribution
```

- [ ] **Step 5: Run the tests**

Run (from the project root, so `backend` resolves as a package): `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -m pytest backend/tests/test_predict.py -v`

Expected: 6 passed. If `ModuleNotFoundError: No module named 'backend'`, run from the project root (not from inside `backend/`) — `python -m pytest` adds the current directory to `sys.path`.

- [ ] **Step 6: Manually start the server and confirm it boots**

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi uvicorn backend.app:app --port 8000`

Expected: server starts without error (registry + encoders load at startup — takes a few seconds), listening on port 8000. Stop it (Ctrl+C) before moving on — Task 6 will restart it alongside the frontend.

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/tests/__init__.py backend/tests/test_predict.py
git commit -m "feat: add FastAPI /predict endpoint with tests"
```

---

### Task 6: `frontend/` — minimal Phase 1 UI

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/next.config.js`
- Create: `frontend/pages/_app.js`
- Create: `frontend/pages/index.js`

**Interfaces:**
- Consumes: Task 5's `POST http://localhost:8000/predict` (same JSON shape documented in Task 5's Interfaces block).
- Produces: a running page at `http://localhost:3000` — nothing downstream in this plan consumes frontend code directly (Task 9 replaces `pages/index.js`'s content, not its role).

- [ ] **Step 1: Write `frontend/package.json`**

```json
{
  "name": "mmi-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  }
}
```

- [ ] **Step 2: Write `frontend/next.config.js`**

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};
module.exports = nextConfig;
```

- [ ] **Step 3: Write `frontend/pages/_app.js`**

```js
export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />;
}
```

- [ ] **Step 4: Write `frontend/pages/index.js`**

```jsx
import { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

// Phase 1: only these 2 models are exported/served. Task 9 (Phase 2)
// widens this to all 8 -- see
// docs/superpowers/specs/2026-08-06-multimodal-intent-app-design.md.
const MODEL_CHOICES = [
  { id: "T", label: "Text-only (M5, accuracy 0.421)" },
  { id: "MISA", label: "MISA text+audio+video fusion (M3, accuracy 0.333)" },
];

export default function Home() {
  const [modelChoice, setModelChoice] = useState("T");
  const [text, setText] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("model_choice", modelChoice);
    if (text) formData.append("text", text);
    if (videoFile) formData.append("video", videoFile);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json();
      if (!response.ok) {
        setError(body.error || "Request failed.");
      } else {
        setResult(body);
      }
    } catch (err) {
      setError(`Could not reach the backend at ${BACKEND_URL}. Is it running?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Multimodal Intent Inference</h1>
      <p>
        This pipeline has no speech-to-text: type what was said yourself,
        even for models that also use audio/video.
      </p>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 12 }}>
          <label>
            Model:{" "}
            <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
              {MODEL_CHOICES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Text (what was said):
            <br />
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              style={{ width: "100%" }}
            />
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Video file (mp4/webm):
            <br />
            <input
              type="file"
              accept="video/mp4,video/webm"
              onChange={(e) => setVideoFile(e.target.files[0] || null)}
            />
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict intent"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 20, color: "#b00020" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p>
            <strong>Predicted intent:</strong> {result.predicted_intent}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
          </p>
          <p style={{ color: "#666" }}>{result.explanation.calibration_caveat}</p>
          {result.explanation.top_words && (
            <div>
              <strong>Top contributing words:</strong>
              <ul>
                {result.explanation.top_words.map((w) => (
                  <li key={w.word}>
                    {w.word} ({w.weight.toFixed(3)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
```

- [ ] **Step 5: Write `frontend/.gitignore`**

```
node_modules/
.next/
```

- [ ] **Step 6: Install frontend dependencies**

Run: `cd frontend && npm install`

Expected: `node_modules/` created, no errors. (Node v24.16.0 / npm 11.13.0 already installed on this machine.)

- [ ] **Step 7: Manually verify end-to-end in a real browser**

1. In one terminal: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi uvicorn backend.app:app --port 8000`
2. In a second terminal: `cd frontend && npm run dev`
3. Open `http://localhost:3000` in a real browser.
4. Golden path 1: select "Text-only", type a sentence (e.g. "I guess we should get going now"), submit — confirm a real predicted intent, confidence percentage, calibration caveat text, and a non-empty top-words list all render.
5. Golden path 2: select "MISA", type a sentence, upload `data/dialogue/raw_clips/S04/E04/374.mp4` as the video file, submit — confirm a real predicted intent and confidence render, and no top-words section appears (MISA has none).
6. Error case: select "MISA", type nothing, upload no file, submit — confirm a clear error message renders (not a blank page or a raw stack trace).
7. Stop both servers (Ctrl+C in each terminal) when done.

This step must actually be performed, not just claimed — per this project's global CLAUDE.md UI-testing rule (start the dev server, exercise the golden path and an error case in a real browser before reporting complete).

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/next.config.js frontend/pages/_app.js frontend/pages/index.js frontend/.gitignore
git commit -m "feat: add minimal Phase 1 frontend"
```

`npm install` will have generated `frontend/node_modules/` and `frontend/package-lock.json`. `frontend/.gitignore` (Step 5) keeps `node_modules/`/`.next/` out of git — `package-lock.json` SHOULD be committed (it pins exact dependency versions).

---

## Phase 2: widen to all 8 models and the full designed UI

**Phase 1 is a complete, working app on its own — everything below is additive, not a rework**, per the design spec's explicit phasing goal.

### Task 7: export the remaining 6 of M5's modality combinations

**Files:**
- Modify: `scripts/export_checkpoints.py`

**Interfaces:**
- Consumes: same as Task 1 (`src/modality_ablation.py`'s functions), unchanged.
- Produces: `models/lr_A.joblib`, `models/lr_V.joblib`, `models/lr_TA.joblib`, `models/lr_TV.joblib`, `models/lr_AV.joblib`, `models/lr_TAV.joblib` (same bundle shape as `models/lr_T.joblib` from Task 1). Task 8 loads all of these.

- [ ] **Step 1: Extend the `EXPECTED` dict and `main()` in `scripts/export_checkpoints.py`**

Update `EXPECTED` (add the 6 new combos' real numbers from this plan's Global Constraints table):

```python
EXPECTED = {
    "T": {"accuracy": 0.4206, "macro_f1": 0.2941},
    "A": {"accuracy": 0.1495, "macro_f1": 0.0687},
    "V": {"accuracy": 0.1121, "macro_f1": 0.0361},
    "TA": {"accuracy": 0.4112, "macro_f1": 0.2466},
    "TV": {"accuracy": 0.3364, "macro_f1": 0.2537},
    "AV": {"accuracy": 0.1963, "macro_f1": 0.0959},
    "TAV": {"accuracy": 0.3458, "macro_f1": 0.2083},
    "MISA": {"accuracy": 0.333, "macro_f1": 0.295},
}
```

Update `main()`:

```python
def main():
    for combo in ["T", "A", "V", "TA", "TV", "AV", "TAV"]:
        export_lr_combo(combo)
    export_misa()
    print("=" * 70)
    print("DONE. All 8 checkpoints verified and saved to models/.")
    print("=" * 70)
```

- [ ] **Step 2: Re-run it**

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python scripts/export_checkpoints.py`

Expected: 8 "Determinism check passed" lines (7 combos + MISA), 8 files present in `models/` (`lr_T.joblib`, `lr_A.joblib`, `lr_V.joblib`, `lr_TA.joblib`, `lr_TV.joblib`, `lr_AV.joblib`, `lr_TAV.joblib`, `misa.pt`).

- [ ] **Step 3: Commit**

```bash
git add scripts/export_checkpoints.py
git commit -m "feat: export remaining 6 modality combinations for Phase 2"
```

---

### Task 8: widen the backend to serve all 8 models

**Files:**
- Modify: `backend/registry.py`
- Modify: `backend/tests/test_predict.py`

**Interfaces:**
- Consumes: Task 7's 6 new checkpoint files.
- Produces: `load_registry()` now returns all 8 keys. No signature changes to anything — this is the "additive, not a rework" property the design spec called for: `backend/app.py`, `backend/inference.py`, and `backend/explanation.py` need zero changes, since they were already written generically over whatever `registry` contains.

- [ ] **Step 1: Extend `LR_COMBOS` in `backend/registry.py`**

```python
LR_COMBOS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
```

(Remove the "Phase 1: only 'T'..." comment above it, or update it to note Phase 2 is now active.)

- [ ] **Step 2: Add tests for 2 more combos to `backend/tests/test_predict.py`**

Add these two test functions (covering a non-text combo and a three-way combo, to prove the registry/inference genuinely generalizes rather than only working for the two Phase 1 cases):

```python
def test_v_combo_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "V"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "V"
    assert isinstance(body["predicted_intent"], str)
    assert body["explanation"]["top_words"] is None  # no text in this combo


def test_tav_combo_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "TAV", "text": "I guess we should get going now"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "TAV"
    assert isinstance(body["predicted_intent"], str)
    assert body["explanation"]["top_words"] is not None
```

- [ ] **Step 3: Run the full test suite**

Run: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -m pytest backend/tests/test_predict.py -v`

Expected: 8 passed (the original 6 from Task 5, plus these 2).

- [ ] **Step 4: Commit**

```bash
git add backend/registry.py backend/tests/test_predict.py
git commit -m "feat: widen backend to serve all 8 modality combinations"
```

---

### Task 9: full 8-model frontend picker with `chatgpt-design` styling

**Files:**
- Modify: `frontend/pages/index.js`
- Create: `frontend/styles/globals.css`
- Modify: `frontend/pages/_app.js`

**Interfaces:**
- Consumes: Task 8's now-8-option backend, plus the design tokens already extracted into `chatgpt-design/DESIGN.md` (or `chatgpt-design/references/DESIGN.md`) at the project root by the `skillui` tool during this app's design phase — read that file directly before writing any CSS.
- Produces: the same page, same `POST /predict` contract, richer UI and full 8-model picker.

- [ ] **Step 1: Read the design tokens**

Before writing any CSS, read `chatgpt-design/DESIGN.md` (check `chatgpt-design/references/DESIGN.md` if the top-level file isn't present) at the project root. Use its actual documented tokens: light theme, OpenAI Sans font stack, 4px spacing grid, colors `background #fcfcfc`, `surface #ececec`, `text-primary #0d0d0d`, `text-muted #8f8f8f`, `border #5d5d5d`, `accent #3a83f7`; border-radius scale `8px, 10px, 12px, 16px, 28px`; explicitly no gradients, no blur, per its documented anti-patterns.

- [ ] **Step 2: Write `frontend/styles/globals.css`**

```css
:root {
  --bg-primary: #fcfcfc;
  --bg-surface: #ececec;
  --text-primary: #0d0d0d;
  --text-muted: #8f8f8f;
  --border: #5d5d5d;
  --accent: #3a83f7;
  --radius: 12px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: "OpenAI Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: var(--radius);
  padding: 8px 16px;
  font-size: 16px;
  cursor: pointer;
}

button:disabled {
  opacity: 0.6;
  cursor: default;
}

select,
textarea,
input[type="file"] {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px;
  font-size: 16px;
  font-family: inherit;
}

.muted {
  color: var(--text-muted);
}

.error {
  color: #b00020;
}
```

- [ ] **Step 3: Import the stylesheet in `frontend/pages/_app.js`**

```js
import "../styles/globals.css";

export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />;
}
```

- [ ] **Step 4: Rewrite `frontend/pages/index.js` with the full 8-model picker**

Replace the `MODEL_CHOICES` constant with all 8 (real accuracy numbers from this plan's Global Constraints table), and restyle the JSX to use the `card`/`muted`/`error` classes from `globals.css` instead of inline styles (keep all existing behavior — form state, submit handler, error handling — identical to Task 6's version):

```jsx
import { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

const MODEL_CHOICES = [
  { id: "T", label: "Text only (M5) — accuracy 0.421" },
  { id: "TA", label: "Text + Audio (M5) — accuracy 0.411" },
  { id: "TV", label: "Text + Video (M5) — accuracy 0.336" },
  { id: "TAV", label: "Text + Audio + Video, concatenated (M5) — accuracy 0.346" },
  { id: "MISA", label: "Text + Audio + Video, MISA fusion (M3) — accuracy 0.333" },
  { id: "AV", label: "Audio + Video, no text (M5) — accuracy 0.196" },
  { id: "A", label: "Audio only (M5) — accuracy 0.150" },
  { id: "V", label: "Video only (M5) — accuracy 0.112" },
];

const NEEDS_TEXT = new Set(["T", "TA", "TV", "TAV", "MISA"]);
const NEEDS_VIDEO = new Set(["A", "V", "TA", "TV", "AV", "TAV", "MISA"]);

export default function Home() {
  const [modelChoice, setModelChoice] = useState("T");
  const [text, setText] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("model_choice", modelChoice);
    if (text) formData.append("text", text);
    if (videoFile) formData.append("video", videoFile);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json();
      if (!response.ok) {
        setError(body.error || "Request failed.");
      } else {
        setResult(body);
      }
    } catch (err) {
      setError(`Could not reach the backend at ${BACKEND_URL}. Is it running?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 640, margin: "48px auto", padding: "0 16px" }}>
      <h1>Multimodal Intent Inference</h1>
      <p className="muted">
        Every model here is honestly labeled with its own real, measured
        accuracy (Milestones 3 and 5) — text-only wins in every controlled
        comparison this project has run so far. This pipeline has no
        speech-to-text: type what was said yourself, even for models that
        also use audio/video.
      </p>

      <form onSubmit={handleSubmit} className="card" style={{ marginTop: 16 }}>
        <div style={{ marginBottom: 12 }}>
          <label>
            Model:{" "}
            <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
              {MODEL_CHOICES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {NEEDS_TEXT.has(modelChoice) && (
          <div style={{ marginBottom: 12 }}>
            <label>
              Text (what was said):
              <br />
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                style={{ width: "100%" }}
              />
            </label>
          </div>
        )}

        {NEEDS_VIDEO.has(modelChoice) && (
          <div style={{ marginBottom: 12 }}>
            <label>
              Video file (mp4/webm):
              <br />
              <input
                type="file"
                accept="video/mp4,video/webm"
                onChange={(e) => setVideoFile(e.target.files[0] || null)}
              />
            </label>
          </div>
        )}

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict intent"}
        </button>
      </form>

      {error && (
        <div className="error" style={{ marginTop: 20 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="card" style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p>
            <strong>Predicted intent:</strong> {result.predicted_intent}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
          </p>
          <p className="muted">{result.explanation.calibration_caveat}</p>
          {result.explanation.top_words && (
            <div>
              <strong>Top contributing words:</strong>
              <ul>
                {result.explanation.top_words.map((w) => (
                  <li key={w.word}>
                    {w.word} ({w.weight.toFixed(3)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
```

Note: the form now conditionally shows only the text box and/or file input each model actually needs (`NEEDS_TEXT`/`NEEDS_VIDEO`), instead of always showing both — this mirrors the backend's own `MODEL_REQUIREMENTS` table (`backend/registry.py`) and prevents a user from filling in an input a given model will never use.

- [ ] **Step 5: Manually verify end-to-end in a real browser**

1. In one terminal: `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi uvicorn backend.app:app --port 8000`
2. In a second terminal: `cd frontend && npm run dev`
3. Open `http://localhost:3000` — confirm the light theme, spacing, and card styling from `chatgpt-design/DESIGN.md`'s tokens are visibly applied (not the plain unstyled Task 6 look).
4. Confirm all 8 models appear in the picker with their real accuracy numbers.
5. Exercise at least 3 different model choices end-to-end (e.g. "T", "V" — confirm the video-only field set appears and the text field is hidden — and "MISA"), confirming each renders a real result.
6. Repeat the Task 6 error case (a model missing a required input) and confirm the error still renders clearly.
7. Stop both servers when done.

This step must actually be performed, not just claimed.

- [ ] **Step 6: Commit**

```bash
git add frontend/pages/index.js frontend/pages/_app.js frontend/styles/globals.css
git commit -m "feat: full 8-model picker with chatgpt-design styling"
```

---

## Final check

After Task 9, the app is fully functional: all 8 models servable, full UI, all backend tests passing, both golden paths and an error case manually verified in a real browser. This is the last task in this plan — after the final whole-branch review is clean, the controller should update `CLAUDE.md` with a new entry describing this app (following the same pattern used for M0-M5), and report completion back to the user per their instruction to not prompt again until finished.
