# Milestone 3 MISA Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement MISA (modality-invariant/-specific fusion) on top of Milestone 2's real audio/video embeddings, and honestly report whether it handles the modality-scale imbalance that hurt concatenation fusion.

**Architecture:** Two new source files — `src/misa_model.py` (the model + losses, importable/testable in isolation) and `src/train_misa.py` (load → split → train-with-early-stopping → evaluate → compare) — plus a narrated notebook. Reuses Milestone 2's already-downloaded 300-clip subset and cached embeddings; zero new data acquisition.

**Tech Stack:** Python 3.11 in the `mmi` conda environment. `torch` (already installed), `pandas`/`numpy`/`scikit-learn`/`matplotlib` (already installed). No new dependencies.

## Global Constraints

- New files only — do not modify any M0/M1/M2 file.
- Reuse `data/mintrec_multimodal/index.csv` and `data/mintrec_multimodal/embeddings/*.npz` as-is — no new download, no re-running Milestone 2's acquisition/embedding scripts.
- Follow established code patterns: module docstrings (goal/how-to-run/terms), `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, `"=" * 70` banners, `load_data()`/`split_data()`/`evaluate()` naming, same stratified `train_test_split` (`test_size=0.20` then `0.1875`, both `random_state=42`, stratified on `intent`) as every prior script.
- No pytest suite exists or should be introduced — verification is "run the script/smoke-test, read the printed output, check generated files."
- **Report the real result honestly** — do not adjust hyperparameters or loss weights to chase a particular accuracy number relative to Milestone 2's 0.117 or the text-only ablation's 0.383. Whatever MISA actually achieves on this data is the finding.
- This is real gradient-based training (a first for this project) — training must be fast (well under a minute, per the design spec's small hidden_dim=128 / ~195 training rows) so it doesn't need a separate confirmation gate; if any single run unexpectedly takes several minutes, that's worth noting in the implementer's report as a deviation from expectation, not silently accepted.
- The project's `commit` skill (secret scan, draft message, staged-files review) is used for every commit, not raw `git commit`.
- Working directly on `master`, no worktree (matches all prior plans in this project).

---

## Task 1: MISA model and losses

**Files:**
- Create: `src/misa_model.py`

**Interfaces:**
- Produces: `MISAModel(text_dim, audio_dim=768, video_dim=512, hidden_dim=128, num_classes=20)`, a `torch.nn.Module` whose `forward(text_feat, audio_feat, video_feat)` returns `(logits, reps)` — `logits` shape `(batch, num_classes)`, `reps` a dict with keys `enc_t, enc_a, enc_v, shared_t, shared_a, shared_v, private_t, private_a, private_v, recon_t, recon_a, recon_v`, each shape `(batch, hidden_dim)`.
- Produces: `misa_loss(logits, labels, reps, sim_weight=0.5, diff_weight=0.5, recon_weight=0.5)` returning `(total_loss: torch.Tensor, parts: dict[str, float])` where `parts` has keys `task, sim, diff, recon`.

- [ ] **Step 1: Write `src/misa_model.py`**

```python
"""
Milestone 3: MISA (Modality-Invariant and -Specific Representations) model.
=================================================================
GOAL OF THIS FILE
    Defines the MISA architecture and its four training losses, kept
    separate from the training loop (src/train_misa.py) so the model
    itself is easy to read without training-loop noise.

    MISA's core idea: instead of just concatenating text/audio/video
    features (Milestone 2's approach, which let audio/video's much larger
    raw magnitude dominate the classifier), split each modality into:
      - a SHARED ("modality-invariant") representation -- the same
        projection is applied to every modality, encouraging them to
        learn a common, comparable representation
      - a PRIVATE ("modality-specific") representation -- a separate
        projection per modality, capturing what's unique to that modality

    Three auxiliary losses shape these representations during training:
      - similarity loss: pulls the three modalities' SHARED
        representations toward each other (they should agree)
      - difference loss: pushes each modality's PRIVATE representation
        away from its own SHARED representation, and away from other
        modalities' PRIVATE representations (they should capture
        different information, not duplicate it)
      - reconstruction loss: shared+private together should be able to
        reconstruct the original encoded representation (a sanity check
        that splitting into shared/private didn't throw information away)

    This is a deliberately simplified adaptation of the original MISA
    paper (Hazarika et al. 2020) -- see
    docs/superpowers/specs/2026-08-06-milestone3-misa-fusion-design.md
    for exactly what was simplified and why (MLP encoders instead of
    BiLSTMs, since Milestone 2 already pooled each clip into one vector
    per modality; simpler similarity/difference losses than the paper's
    CMD/Frobenius-norm versions).

TERMS YOU'LL SEE
    - embedding / encoded representation : a fixed-length vector a small
      neural network produces to summarize its input
    - shared subspace / private subspace : two different "rooms" a
      modality's information gets projected into -- shared is meant to
      hold what's common across modalities, private what's unique
    - self-attention : a mechanism where each item in a set can look at
      (and be influenced by) every other item before deciding its output
    - orthogonal : at a 90-degree angle -- vectors that are orthogonal
      share no common direction, i.e. capture unrelated information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MISAModel(nn.Module):
    def __init__(self, text_dim, audio_dim=768, video_dim=512, hidden_dim=128, num_classes=20):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_encoder = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU())
        self.audio_encoder = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU())
        self.video_encoder = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.ReLU())

        # SAME projection for every modality -- this weight-sharing is what
        # makes the result "modality-invariant."
        self.shared_proj = nn.Linear(hidden_dim, hidden_dim)

        # A SEPARATE projection per modality -- this is what makes each
        # result "modality-specific."
        self.private_text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.private_audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.private_video_proj = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(2 * hidden_dim, hidden_dim)

        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=2 * hidden_dim, batch_first=True
        )
        self.fusion_layer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_feat, audio_feat, video_feat):
        """
        text_feat, audio_feat, video_feat: (batch, text_dim)/(batch, 768)/(batch, 512)
        Returns: logits (batch, num_classes), and a dict of intermediate
        representations the loss functions need.
        """
        enc_t = self.text_encoder(text_feat)
        enc_a = self.audio_encoder(audio_feat)
        enc_v = self.video_encoder(video_feat)

        shared_t = self.shared_proj(enc_t)
        shared_a = self.shared_proj(enc_a)
        shared_v = self.shared_proj(enc_v)

        private_t = self.private_text_proj(enc_t)
        private_a = self.private_audio_proj(enc_a)
        private_v = self.private_video_proj(enc_v)

        recon_t = self.decoder(torch.cat([shared_t, private_t], dim=-1))
        recon_a = self.decoder(torch.cat([shared_a, private_a], dim=-1))
        recon_v = self.decoder(torch.cat([shared_v, private_v], dim=-1))

        # Stack the 6 representations as a length-6 sequence per sample so
        # self-attention can let them influence each other, then mean-pool.
        fusion_input = torch.stack(
            [shared_t, shared_a, shared_v, private_t, private_a, private_v], dim=1
        )  # (batch, 6, hidden_dim)
        fused = self.fusion_layer(fusion_input).mean(dim=1)  # (batch, hidden_dim)

        logits = self.classifier(fused)

        reps = {
            "enc_t": enc_t, "enc_a": enc_a, "enc_v": enc_v,
            "shared_t": shared_t, "shared_a": shared_a, "shared_v": shared_v,
            "private_t": private_t, "private_a": private_a, "private_v": private_v,
            "recon_t": recon_t, "recon_a": recon_a, "recon_v": recon_v,
        }
        return logits, reps


def similarity_loss(reps):
    """Pull the three modalities' SHARED representations toward each other."""
    s_t, s_a, s_v = reps["shared_t"], reps["shared_a"], reps["shared_v"]
    return (
        F.mse_loss(s_t, s_a) + F.mse_loss(s_t, s_v) + F.mse_loss(s_a, s_v)
    ) / 3.0


def difference_loss(reps):
    """Push each modality's PRIVATE representation away from its own SHARED
    representation, and away from other modalities' PRIVATE representations."""

    def cos_sq(a, b):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return (a_norm * b_norm).sum(dim=-1).pow(2).mean()

    private_shared = (
        cos_sq(reps["private_t"], reps["shared_t"])
        + cos_sq(reps["private_a"], reps["shared_a"])
        + cos_sq(reps["private_v"], reps["shared_v"])
    ) / 3.0
    private_private = (
        cos_sq(reps["private_t"], reps["private_a"])
        + cos_sq(reps["private_t"], reps["private_v"])
        + cos_sq(reps["private_a"], reps["private_v"])
    ) / 3.0
    return private_shared + private_private


def reconstruction_loss(reps):
    """Shared+private together should reconstruct the original encoded
    representation -- a sanity check that no information was thrown away."""
    return (
        F.mse_loss(reps["recon_t"], reps["enc_t"].detach())
        + F.mse_loss(reps["recon_a"], reps["enc_a"].detach())
        + F.mse_loss(reps["recon_v"], reps["enc_v"].detach())
    ) / 3.0


def misa_loss(logits, labels, reps, sim_weight=0.5, diff_weight=0.5, recon_weight=0.5):
    """Combine task loss with the three MISA auxiliary losses."""
    task = F.cross_entropy(logits, labels)
    sim = similarity_loss(reps)
    diff = difference_loss(reps)
    recon = reconstruction_loss(reps)
    total = task + sim_weight * sim + diff_weight * diff + recon_weight * recon
    return total, {"task": task.item(), "sim": sim.item(), "diff": diff.item(), "recon": recon.item()}


if __name__ == "__main__":
    # Smoke test: random tensors of the right shapes should produce
    # correctly-shaped logits and finite losses. Run with:
    #   python src/misa_model.py
    torch.manual_seed(0)
    batch, text_dim, num_classes = 4, 1400, 20
    model = MISAModel(text_dim=text_dim, num_classes=num_classes)

    text_feat = torch.randn(batch, text_dim)
    audio_feat = torch.randn(batch, 768)
    video_feat = torch.randn(batch, 512)
    labels = torch.randint(0, num_classes, (batch,))

    logits, reps = model(text_feat, audio_feat, video_feat)
    assert logits.shape == (batch, num_classes), f"unexpected logits shape {logits.shape}"

    total_loss, parts = misa_loss(logits, labels, reps)
    assert torch.isfinite(total_loss), "loss is not finite"
    for name, value in parts.items():
        assert value == value, f"{name} loss is NaN"  # NaN != NaN

    print("Smoke test passed.")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss breakdown: {parts}, total={total_loss.item():.4f}")
```

- [ ] **Step 2: Run the smoke test**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/misa_model.py
```

Expected: `Smoke test passed.` followed by `logits shape: (4, 20)` and a loss breakdown dict with four finite (non-NaN) float values, plus a finite `total`.

- [ ] **Step 3: Commit**

Use the `commit` skill. Stage `src/misa_model.py` explicitly. Suggested message subject: `feat: add MISA model architecture for Milestone 3`.

---

## Task 2: MISA training script

**Files:**
- Create: `src/train_misa.py`

**Interfaces:**
- Consumes: `data/mintrec_multimodal/index.csv` and `data/mintrec_multimodal/embeddings/*.npz` (from Milestone 2, already exist). Consumes `MISAModel` and `misa_loss` from `src/misa_model.py` (Task 1) via `from misa_model import MISAModel, misa_loss` (works because Python adds the running script's own directory to `sys.path` automatically — no package setup needed, matches how this project's other `src/*.py` scripts are run).
- Produces: `results/misa_predictions.csv` (columns `text, true, predicted`) and `results/misa_confusion_matrix.csv` (20x20 labeled).

- [ ] **Step 1: Write `src/train_misa.py`**

```python
"""
Milestone 3: train and evaluate the MISA fusion model on real MIntRec data.
=================================================================
GOAL OF THIS FILE
    Same overall shape as Milestone 2's scripts (load -> split -> train ->
    evaluate), but the "train" step is now REAL gradient-based training of
    a small neural network (src/misa_model.py), not just fitting a linear
    classifier on frozen features. This is the first script in the project
    where the `val` split actually gets used for something (early
    stopping), instead of just being computed and reserved.

HOW TO RUN
    conda activate mmi
    python src/misa_model.py       # smoke test, optional but recommended first
    python src/train_misa.py

    Reuses Milestone 2's already-downloaded data and embeddings -- no new
    download. Training is small (hidden_dim=128, ~195 training rows) and
    should finish in well under a minute on CPU.

TERMS YOU'LL SEE (new ones, beyond Milestones 0-2)
    - epoch          : one full pass through the training data
    - early stopping : stop training once validation performance stops
                        improving, instead of training a fixed number of
                        epochs -- prevents overfitting and saves time
    - patience        : how many epochs to keep waiting for improvement
                        before actually stopping
    - checkpoint      : a saved copy of the model's weights at a specific
                        point in training (here: whenever validation
                        macro-F1 hits a new best)
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from misa_model import MISAModel, misa_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "mintrec_multimodal" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

HIDDEN_DIM = 128
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
SEED = 0


def load_data():
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
    return df


def split_data(df):
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
    print(f"val:   {len(val)} samples  (used for early stopping -- first time this project uses it!)")
    print(f"test:  {len(test)} samples  (final honest judgement)")
    return train, val, test


def build_audio_features(df):
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["audio"] for sid in df["sample_id"]])


def build_video_features(df):
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["video"] for sid in df["sample_id"]])


def featurize(train, val, test):
    """TF-IDF (fit on train only) + raw audio/video embeddings -- no
    StandardScaler here: MISA's per-modality MLP encoders learn
    scale-appropriate weights themselves, which is part of the point."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    text_train = tfidf.fit_transform(train["text"]).toarray().astype(np.float32)
    text_val = tfidf.transform(val["text"]).toarray().astype(np.float32)
    text_test = tfidf.transform(test["text"]).toarray().astype(np.float32)

    audio_train = build_audio_features(train).astype(np.float32)
    audio_val = build_audio_features(val).astype(np.float32)
    audio_test = build_audio_features(test).astype(np.float32)

    video_train = build_video_features(train).astype(np.float32)
    video_val = build_video_features(val).astype(np.float32)
    video_test = build_video_features(test).astype(np.float32)

    return (
        (text_train, audio_train, video_train),
        (text_val, audio_val, video_val),
        (text_test, audio_test, video_test),
        tfidf,
    )


def to_tensors(text, audio, video, labels, label2id):
    return (
        torch.from_numpy(text),
        torch.from_numpy(audio),
        torch.from_numpy(video),
        torch.tensor([label2id[l] for l in labels], dtype=torch.long),
    )


def run_epoch(model, optimizer, text, audio, video, labels, batch_size, train_mode):
    model.train(train_mode)
    n = text.shape[0]
    indices = torch.randperm(n) if train_mode else torch.arange(n)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        bt, ba, bv, bl = text[batch_idx], audio[batch_idx], video[batch_idx], labels[batch_idx]

        if train_mode:
            optimizer.zero_grad()
            logits, reps = model(bt, ba, bv)
            loss, _ = misa_loss(logits, bl, reps)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, reps = model(bt, ba, bv)
                loss, _ = misa_loss(logits, bl, reps)

        total_loss += loss.item() * len(batch_idx)
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(bl.tolist())

    avg_loss = total_loss / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_misa(train_tensors, val_tensors):
    print("\n" + "=" * 70)
    print("STEP 3  TRAIN THE MISA MODEL (with early stopping on validation macro-F1)")
    print("=" * 70)
    text_train, audio_train, video_train, labels_train = train_tensors
    text_val, audio_val, video_val, labels_val = val_tensors

    torch.manual_seed(SEED)
    model = MISAModel(text_dim=text_train.shape[1], num_classes=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(
            model, optimizer, text_train, audio_train, video_train, labels_train, BATCH_SIZE, train_mode=True
        )
        val_loss, val_f1 = run_epoch(
            model, optimizer, text_val, audio_val, video_val, labels_val, BATCH_SIZE, train_mode=False
        )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
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


def evaluate(model, test_tensors, id2label, test_df):
    print("\n" + "=" * 70)
    print("STEP 4  EVALUATE ON THE TEST SET")
    print("=" * 70)
    text_test, audio_test, video_test, labels_test = test_tensors

    model.eval()
    with torch.no_grad():
        logits, _ = model(text_test, audio_test, video_test)
    preds_id = logits.argmax(dim=1).tolist()
    preds = [id2label[p] for p in preds_id]
    truth = [id2label[l] for l in labels_test.tolist()]

    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro", zero_division=0)
    print(f"Accuracy : {acc:.3f}   (share of predictions that were exactly right)")
    print(f"Macro-F1 : {macro_f1:.3f}   (fairer average across all intents)")

    print("\nPer-class report (precision/recall/F1 for each intent):")
    print(classification_report(truth, preds, zero_division=0))

    RESULTS_DIR.mkdir(exist_ok=True)
    labels_sorted = sorted(set(truth) | set(preds))
    cm = confusion_matrix(truth, preds, labels=labels_sorted)
    cm_path = RESULTS_DIR / "misa_confusion_matrix.csv"
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(cm_path)
    print(f"Confusion matrix (20x20, too wide for terminal) saved to {cm_path.relative_to(PROJECT_ROOT)}")

    out = RESULTS_DIR / "misa_predictions.csv"
    pd.DataFrame({"text": test_df["text"].tolist(), "true": truth, "predicted": preds}).to_csv(out, index=False)
    print(f"Saved every test prediction to {out.relative_to(PROJECT_ROOT)}")

    return acc, macro_f1


def main():
    df = load_data()
    train, val, test = split_data(df)

    (text_train, audio_train, video_train), (text_val, audio_val, video_val), \
        (text_test, audio_test, video_test), tfidf = featurize(train, val, test)

    labels_sorted = sorted(df["intent"].unique())
    label2id = {name: i for i, name in enumerate(labels_sorted)}
    id2label = {i: name for name, i in label2id.items()}

    train_tensors = to_tensors(text_train, audio_train, video_train, train["intent"].tolist(), label2id)
    val_tensors = to_tensors(text_val, audio_val, video_val, val["intent"].tolist(), label2id)
    test_tensors = to_tensors(text_test, audio_test, video_test, test["intent"].tolist(), label2id)

    model = train_misa(train_tensors, val_tensors)
    acc, macro_f1 = evaluate(model, test_tensors, id2label, test)

    print("\n" + "=" * 70)
    print("DONE. Compare this to Milestone 2's concatenation-fusion result:")
    print("  Concatenation fusion (M2): accuracy 0.117, macro-F1 0.096")
    print("  Text-only in this pipeline (M2 ablation): accuracy 0.383")
    print(f"  MISA (M3):                 accuracy {acc:.3f}, macro-F1 {macro_f1:.3f}")
    print("Ask Claude Code: 'Did the shared/private decomposition actually help")
    print("here, or does modality imbalance still win?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/train_misa.py
```

Expected: STEP 1-4 banners; per-epoch progress lines showing train/val loss and macro-F1 (with `*` marking new best-val epochs); either an early-stopping message or a max-epochs-reached message; "Restored best checkpoint"; `Accuracy`/`Macro-F1` lines (the real, previously-unknown result); a classification report; confirmation the confusion matrix and predictions were saved; the closing comparison against Milestone 2's 0.117. Should finish in well under a minute — if it takes several minutes, note this as a deviation in your report rather than silently accepting it.

- [ ] **Step 3: Verify training actually happened and results are self-consistent**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd
from sklearn.metrics import accuracy_score

preds = pd.read_csv('results/misa_predictions.csv')
assert len(preds) == 60, f'expected 60 test rows (20% of 300), got {len(preds)}'
assert list(preds.columns) == ['text', 'true', 'predicted']
acc = accuracy_score(preds['true'], preds['predicted'])
print(f'accuracy from saved predictions: {acc:.3f}')

cm = pd.read_csv('results/misa_confusion_matrix.csv', index_col=0)
assert cm.values.sum() == 60, f'confusion matrix should sum to 60, got {cm.values.sum()}'
print('OK: results files are internally consistent.')
"
```

Expected: `OK: results files are internally consistent.` No accuracy floor assertion — report whatever the real number is, per the Global Constraints.

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/train_misa.py` explicitly — not `results/` (gitignored). Suggested message subject: `feat: add MISA training script for Milestone 3`.

---

## Task 3: Tutorial notebook

**Files:**
- Create: `notebooks/milestone3_misa_fusion.ipynb`
- Create (scratchpad only, not committed): a Python builder script filling the notebook's cells, same throwaway pattern used for every prior notebook in this project.

**Interfaces:**
- Consumes: `data/mintrec_multimodal/index.csv` + `embeddings/*.npz` (Milestone 2) and imports `MISAModel`/`misa_loss` from `src/misa_model.py` (Task 1) — this notebook IMPORTS the model class rather than re-implementing it inline (unlike M0-M2's notebooks, which duplicate their scripts' simpler logic). The model architecture is complex enough that duplicating it would risk drift from the actual reviewed implementation; importing keeps one source of truth. Requires adding `src/` to `sys.path` at the top of the notebook.

- [ ] **Step 1: Scaffold the notebook from the tutorial template**

```bash
python "C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py" --kind tutorial --title "Milestone 3 - MISA Fusion" --out "notebooks/milestone3_misa_fusion.ipynb"
```

Expected: `Wrote ...notebooks\milestone3_misa_fusion.ipynb using kind=tutorial.`

- [ ] **Step 2: Write the cell-filling builder script**

Save to the scratchpad (e.g. `build_m3_misa_notebook.py`), adjusting `NB_PATH` if your scratchpad differs:

```python
"""Builds notebooks/milestone3_misa_fusion.ipynb by filling the scaffolded template."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\dbest\Downloads\multimodal-intent-inference\multimodal-intent-inference\notebooks\milestone3_misa_fusion.ipynb")


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
        "# Tutorial: Milestone 3 - MISA Fusion",
        "",
        "Audience:",
        "- You've completed Milestone 2 (real audio/video, concatenation fusion) and know that naive",
        "  concatenation let dense audio/video embeddings dominate the classifier, hurting accuracy.",
        "",
        "Prerequisites:",
        "- Run `python src/misa_model.py` (a quick smoke test) and `python src/train_misa.py` at least once first",
        "  -- this notebook imports the model class from `src/misa_model.py` rather than redefining it.",
        "",
        "Learning goals:",
        "- By the end, you can explain what 'shared' vs. 'private' modality representations are, why MISA's",
        "  auxiliary losses exist, and read a training curve (the first one in this project -- everything",
        "  before this milestone trained instantly, with nothing to plot over time).",
        "",
        "**What's new here:** this is the first REAL gradient-based training in the project (a small neural",
        "network, trained with backpropagation) rather than fitting a linear classifier on frozen features.",
        "It's also the first time the `val` split actually gets used (for early stopping) instead of just",
        "being computed and set aside.",
    ),
    md(
        "## Outline",
        "",
        "1. Setup",
        "2. Step 1 - Load the real MIntRec multimodal index (same data as Milestone 2)",
        "3. Step 2 - Split into train / validation / test",
        "4. Step 3 - The MISA architecture, in plain language",
        "5. Step 4 - Train, with early stopping on validation macro-F1",
        "6. Training curves",
        "7. Step 5 - Evaluate, and compare against Milestone 2",
        "8. Look at the mistakes",
        "9. Exercises",
        "10. Pitfalls and extensions",
    ),
    code(
        "import sys",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix",
        "import matplotlib.pyplot as plt",
        "",
        "NOTEBOOK_DIR = Path.cwd()",
        "PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR",
        "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))",
        "from misa_model import MISAModel, misa_loss  # the reviewed model class -- not redefined here",
        "",
        "INDEX_PATH = PROJECT_ROOT / \"data\" / \"mintrec_multimodal\" / \"index.csv\"",
        "EMBEDDINGS_DIR = PROJECT_ROOT / \"data\" / \"mintrec_multimodal\" / \"embeddings\"",
        "RESULTS_DIR = PROJECT_ROOT / \"results\"",
        "INDEX_PATH",
    ),
    md(
        "## Step 1 - Load the real MIntRec multimodal index",
        "",
        "Identical data source to Milestone 2 -- the same 300-clip subset, same embeddings. Only the fusion",
        "method changes in this notebook.",
    ),
    code(
        "df = pd.read_csv(INDEX_PATH)",
        "print(f\"Loaded {len(df)} samples from {INDEX_PATH.name}\")",
        "df[\"intent\"].value_counts()",
    ),
    md(
        "## Step 2 - Split into train / validation / test",
        "",
        "Same split as every prior script -- but this time `val` actually gets used, for early stopping below.",
    ),
    code(
        "train_val, test = train_test_split(",
        "    df, test_size=0.20, random_state=42, stratify=df[\"intent\"]",
        ")",
        "train, val = train_test_split(",
        "    train_val, test_size=0.1875, random_state=42, stratify=train_val[\"intent\"]",
        ")",
        "print(f\"train: {len(train)} samples\")",
        "print(f\"val:   {len(val)} samples  (used for early stopping this time)\")",
        "print(f\"test:  {len(test)} samples\")",
    ),
    md(
        "## Step 3 - The MISA architecture, in plain language",
        "",
        "Milestone 2's fusion was `np.hstack([text, audio, video])` -- glue the three feature vectors end to end.",
        "That let audio/video's much larger raw magnitude dominate the classifier even after scaling.",
        "",
        "MISA instead learns, for each modality, two separate representations:",
        "- **Shared** (\"modality-invariant\"): the exact same projection (same weights) is applied to text,",
        "  audio, and video -- encouraging all three to land in a common, comparable space.",
        "- **Private** (\"modality-specific\"): a separate projection per modality, capturing what's unique to it.",
        "",
        "Three extra loss terms (beyond the usual classification loss) shape these during training:",
        "- **Similarity loss** -- pulls the three SHARED representations toward each other.",
        "- **Difference loss** -- pushes each PRIVATE representation away from its own SHARED representation",
        "  and away from other modalities' PRIVATE representations, so they don't just duplicate each other.",
        "- **Reconstruction loss** -- shared+private together must be able to reconstruct the original encoded",
        "  representation, checking that splitting into shared/private didn't throw information away.",
        "",
        "Finally, all six representations (3 shared + 3 private) go through one self-attention layer -- so each",
        "can be influenced by the others -- before a final linear classifier.",
        "",
        "See `src/misa_model.py` for the full implementation (imported above, not re-shown here since the",
        "reviewed, tested version already lives in one place).",
    ),
    code(
        "# Peek at the model's structure and size.",
        "model_preview = MISAModel(text_dim=1000, num_classes=20)  # text_dim is a placeholder just to inspect shape",
        "n_params = sum(p.numel() for p in model_preview.parameters())",
        "print(model_preview)",
        "print(f\"\\nTotal trainable parameters: {n_params:,}\")",
    ),
    md(
        "## Step 4 - Train, with early stopping on validation macro-F1",
        "",
        "Unlike every prior milestone (which fit instantly), this is real iterative training: the model sees",
        "the training data many times (epochs), and we watch validation performance to know when to stop --",
        "training longer isn't always better, since the model can start memorizing the training set instead of",
        "learning generalizable patterns.",
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
        "tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)",
        "text_train = tfidf.fit_transform(train[\"text\"]).toarray().astype(np.float32)",
        "text_val = tfidf.transform(val[\"text\"]).toarray().astype(np.float32)",
        "text_test = tfidf.transform(test[\"text\"]).toarray().astype(np.float32)",
        "",
        "audio_train, audio_val, audio_test = (build_audio_features(d).astype(np.float32) for d in (train, val, test))",
        "video_train, video_val, video_test = (build_video_features(d).astype(np.float32) for d in (train, val, test))",
        "",
        "labels_sorted = sorted(df[\"intent\"].unique())",
        "label2id = {name: i for i, name in enumerate(labels_sorted)}",
        "id2label = {i: name for name, i in label2id.items()}",
        "",
        "",
        "def to_tensors(text, audio, video, labels):",
        "    return (",
        "        torch.from_numpy(text), torch.from_numpy(audio), torch.from_numpy(video),",
        "        torch.tensor([label2id[l] for l in labels], dtype=torch.long),",
        "    )",
        "",
        "",
        "text_train_t, audio_train_t, video_train_t, labels_train_t = to_tensors(text_train, audio_train, video_train, train[\"intent\"].tolist())",
        "text_val_t, audio_val_t, video_val_t, labels_val_t = to_tensors(text_val, audio_val, video_val, val[\"intent\"].tolist())",
        "text_test_t, audio_test_t, video_test_t, labels_test_t = to_tensors(text_test, audio_test, video_test, test[\"intent\"].tolist())",
        "print(f\"Fused input: text={text_train.shape[1]}-dim, audio=768-dim, video=512-dim\")",
    ),
    code(
        "import copy",
        "",
        "HIDDEN_DIM, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, SEED = 128, 16, 100, 15, 1e-3, 0",
        "",
        "",
        "def run_epoch(model, optimizer, text, audio, video, labels, batch_size, train_mode):",
        "    model.train(train_mode)",
        "    n = text.shape[0]",
        "    indices = torch.randperm(n) if train_mode else torch.arange(n)",
        "    total_loss = 0.0",
        "    all_preds, all_labels = [], []",
        "    for start in range(0, n, batch_size):",
        "        batch_idx = indices[start:start + batch_size]",
        "        bt, ba, bv, bl = text[batch_idx], audio[batch_idx], video[batch_idx], labels[batch_idx]",
        "        if train_mode:",
        "            optimizer.zero_grad()",
        "            logits, reps = model(bt, ba, bv)",
        "            loss, _ = misa_loss(logits, bl, reps)",
        "            loss.backward()",
        "            optimizer.step()",
        "        else:",
        "            with torch.no_grad():",
        "                logits, reps = model(bt, ba, bv)",
        "                loss, _ = misa_loss(logits, bl, reps)",
        "        total_loss += loss.item() * len(batch_idx)",
        "        all_preds.extend(logits.argmax(dim=1).tolist())",
        "        all_labels.extend(bl.tolist())",
        "    avg_loss = total_loss / n",
        "    macro_f1 = f1_score(all_labels, all_preds, average=\"macro\", zero_division=0)",
        "    return avg_loss, macro_f1",
        "",
        "",
        "torch.manual_seed(SEED)",
        "model = MISAModel(text_dim=text_train.shape[1], num_classes=20)",
        "optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)",
        "",
        "history = {\"train_loss\": [], \"val_loss\": [], \"train_f1\": [], \"val_f1\": []}",
        "best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0",
        "",
        "for epoch in range(1, MAX_EPOCHS + 1):",
        "    train_loss, train_f1 = run_epoch(model, optimizer, text_train_t, audio_train_t, video_train_t, labels_train_t, BATCH_SIZE, True)",
        "    val_loss, val_f1 = run_epoch(model, optimizer, text_val_t, audio_val_t, video_val_t, labels_val_t, BATCH_SIZE, False)",
        "    history[\"train_loss\"].append(train_loss); history[\"val_loss\"].append(val_loss)",
        "    history[\"train_f1\"].append(train_f1); history[\"val_f1\"].append(val_f1)",
        "",
        "    improved = val_f1 > best_val_f1",
        "    if improved:",
        "        best_val_f1, best_state, epochs_without_improvement = val_f1, copy.deepcopy(model.state_dict()), 0",
        "    else:",
        "        epochs_without_improvement += 1",
        "    if epoch <= 5 or epoch % 10 == 0 or improved:",
        "        marker = \" *\" if improved else \"\"",
        "        print(f\"epoch {epoch:3d}  train_loss={train_loss:.4f} train_f1={train_f1:.3f}  val_loss={val_loss:.4f} val_f1={val_f1:.3f}{marker}\")",
        "    if epochs_without_improvement >= PATIENCE:",
        "        print(f\"\\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).\")",
        "        break",
        "else:",
        "    print(f\"\\nReached max epochs ({MAX_EPOCHS}) without early stopping.\")",
        "",
        "model.load_state_dict(best_state)",
        "print(f\"Restored best checkpoint (val macro-F1={best_val_f1:.3f}).\")",
    ),
    md(
        "## Training curves",
        "",
        "This is the first plot of its kind in this project -- everything before Milestone 3 trained instantly,",
        "with no \"progress over time\" to show. Watch for: train loss should generally decrease; val loss",
        "decreasing alongside it is good, val loss going back up while train loss keeps dropping is the",
        "textbook overfitting signature (not necessarily what happens here -- read the actual curve).",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
        "epochs_ran = range(1, len(history[\"train_loss\"]) + 1)",
        "axes[0].plot(epochs_ran, history[\"train_loss\"], label=\"train\")",
        "axes[0].plot(epochs_ran, history[\"val_loss\"], label=\"val\")",
        "axes[0].set_xlabel(\"epoch\"); axes[0].set_ylabel(\"loss\"); axes[0].set_title(\"Loss\"); axes[0].legend()",
        "axes[1].plot(epochs_ran, history[\"train_f1\"], label=\"train\")",
        "axes[1].plot(epochs_ran, history[\"val_f1\"], label=\"val\")",
        "axes[1].set_xlabel(\"epoch\"); axes[1].set_ylabel(\"macro-F1\"); axes[1].set_title(\"Macro-F1\"); axes[1].legend()",
        "fig.tight_layout()",
        "plt.show()",
    ),
    md(
        "## Step 5 - Evaluate, and compare against Milestone 2",
    ),
    code(
        "model.eval()",
        "with torch.no_grad():",
        "    logits, _ = model(text_test_t, audio_test_t, video_test_t)",
        "preds = [id2label[p] for p in logits.argmax(dim=1).tolist()]",
        "truth = [id2label[l] for l in labels_test_t.tolist()]",
        "",
        "acc = accuracy_score(truth, preds)",
        "macro_f1 = f1_score(truth, preds, average=\"macro\", zero_division=0)",
        "print(f\"Accuracy : {acc:.3f}\")",
        "print(f\"Macro-F1 : {macro_f1:.3f}\")",
    ),
    code(
        "print(classification_report(truth, preds, zero_division=0))",
    ),
    code(
        "labels_present = sorted(set(truth) | set(preds))",
        "cm = confusion_matrix(truth, preds, labels=labels_present)",
        "",
        "fig, ax = plt.subplots(figsize=(10, 9))",
        "im = ax.imshow(cm, cmap=\"Blues\")",
        "ax.set_xticks(range(len(labels_present)), labels_present, rotation=90)",
        "ax.set_yticks(range(len(labels_present)), labels_present)",
        "ax.set_xlabel(\"predicted\"); ax.set_ylabel(\"true\")",
        "ax.set_title(\"MISA fusion confusion matrix\")",
        "fig.colorbar(im, ax=ax, shrink=0.8, label=\"count\")",
        "fig.tight_layout()",
        "plt.show()",
    ),
    md(
        "### Compare against Milestone 2's concatenation fusion",
    ),
    code(
        "comparison = pd.DataFrame([",
        "    {\"model\": \"Text-only (M2 ablation, same 300-clip data)\", \"accuracy\": 0.383, \"macro_f1\": float(\"nan\")},",
        "    {\"model\": \"Concatenation fusion (M2)\", \"accuracy\": 0.117, \"macro_f1\": 0.096},",
        "    {\"model\": \"MISA fusion (M3)\", \"accuracy\": acc, \"macro_f1\": macro_f1},",
        "]).set_index(\"model\")",
        "comparison",
    ),
    md(
        "**Interpretation:** write this once you see the real numbers above -- did MISA's shared/private",
        "decomposition recover some or all of the gap between concatenation fusion (0.117) and text-only",
        "(0.383)? If MISA still underperforms text-only, that's still an honest, useful finding: it would mean",
        "modality-scale imbalance isn't the only thing hurting fusion here, or that 195 training rows simply",
        "isn't enough to learn a 6-way shared/private decomposition well. Either result teaches something real.",
    ),
    md(
        "## Look at the mistakes",
    ),
    code(
        "results_df = pd.DataFrame({\"text\": test[\"text\"].tolist(), \"true\": truth, \"predicted\": preds})",
        "mistakes = results_df[results_df[\"true\"] != results_df[\"predicted\"]]",
        "print(f\"{len(mistakes)} of {len(results_df)} test sentences were misclassified ({len(mistakes)/len(results_df):.1%}).\")",
        "mistakes.head(10)",
    ),
    md(
        "## Exercises",
        "",
        "1. Try changing `sim_weight`/`diff_weight`/`recon_weight` in `src/misa_model.py`'s `misa_loss` call",
        "   (edit the training loop above to pass different values) -- does emphasizing the auxiliary losses",
        "   more or less change the result?",
        "2. `HIDDEN_DIM = 128` above. Try a smaller value (e.g. 32) and a larger one (e.g. 256) -- with only 195",
        "   training rows, does a bigger model actually help, or does it just overfit faster?",
    ),
    md(
        "## Pitfalls and extensions",
        "",
        "**Common mistake:** judging a fusion method from a single training run. Neural network training has",
        "real run-to-run randomness (weight initialization, batch shuffling) -- rerunning this notebook with a",
        "different `SEED` could shift the result somewhat. Treat this milestone's number as one data point, not",
        "a definitive verdict on MISA versus concatenation fusion.",
        "",
        "**Extension:** this MISA adaptation uses simple MLP encoders and simplified similarity/difference",
        "losses instead of the original paper's BiLSTM encoders and CMD loss -- see",
        "`docs/superpowers/specs/2026-08-06-milestone3-misa-fusion-design.md` for exactly what was simplified",
        "and why. A natural extension: scale up to the full 2,213-clip dataset (more training data might matter",
        "more for a trainable fusion method than it did for M0-M2's frozen-feature approaches).",
    ),
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells to {NB_PATH}")
```

- [ ] **Step 3: Run the builder script**

```bash
python <path-to-scratchpad>/build_m3_misa_notebook.py
```

Expected: `Wrote 21 cells to ...notebooks\milestone3_misa_fusion.ipynb`.

- [ ] **Step 4: Execute the notebook top-to-bottom**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi jupyter nbconvert --to notebook --execute --inplace "notebooks/milestone3_misa_fusion.ipynb"
```

Expected: `[NbConvertApp] Writing ... bytes to ...notebooks\milestone3_misa_fusion.ipynb` with no Python traceback (the usual symlink/TCP-kernel warnings are expected and harmless). Should finish in well under a minute — actual training happens in this notebook (same small model as the script), so allow it to genuinely complete rather than assuming it's stuck.

- [ ] **Step 5: Verify no errors in the executed notebook**

```bash
python -c "
import json
nb = json.load(open('notebooks/milestone3_misa_fusion.ipynb', encoding='utf-8'))
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

Use the `commit` skill. Stage `notebooks/milestone3_misa_fusion.ipynb` only (the builder script stays in the scratchpad). Suggested message subject: `feat: add Milestone 3 MISA fusion notebook`.

---

## Final check

After Task 3, read back the real accuracy/macro-F1 from Task 2's run and the notebook's comparison table. This plan does not include a `CLAUDE.md` update step — per this project's established pattern (M1, M2), that update is a controller-level decision. Since this milestone is running under autonomous authorization (no per-milestone check-in), update `CLAUDE.md`'s M3 entry directly once this plan's final review is clean, using the real results, then proceed to Milestone 4 without waiting for confirmation.
