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
