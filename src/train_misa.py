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
    model = MISAModel(text_dim=text_train.shape[1], hidden_dim=HIDDEN_DIM, num_classes=20)
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
    print("\nCareful with the credit, though: this script alone cannot tell you WHY")
    print("MISA beats concatenation. Replacing a linear classifier with ANY small")
    print("trainable network that learns its own per-modality scaling would also")
    print("help, with or without the shared/private split. Separating the two needs")
    print("a controlled ablation over several seeds -- Milestone 5's job.")
    print("Ask Claude Code: 'How would I ablate the shared/private split to find out")
    print("how much of this gain it is actually responsible for?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
