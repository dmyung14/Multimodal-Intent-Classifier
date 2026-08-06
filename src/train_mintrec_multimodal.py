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
