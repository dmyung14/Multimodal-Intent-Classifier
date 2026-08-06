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
      frequency is in the hundreds, and RGB values go up to 255.
      StandardScaler rescales every feature to have mean 0 and standard
      deviation 1 first, so no modality's weight in the classifier is
      decided purely by how big its raw numbers happen to be. That's
      usually the right default -- you rarely know in advance which
      modality will turn out to carry the most signal, and without
      scaling, whichever one has the largest raw numbers can dominate for
      no good reason.
      It isn't free, though. On this toy dataset the 6 audio+video
      numbers happen to carry an almost perfect signal (video alone gets
      near-100% accuracy), while the ~480 TF-IDF text dimensions are much
      noisier. StandardScaler treats all of them as equally important by
      construction, which here means diluting a strong, low-dimensional
      signal by averaging it in with a lot of weak, high-dimensional
      noise -- accuracy is measurably better WITHOUT scaling on this
      particular data. We still scale by default below, because "which
      modality is strongest" won't be known that clearly on real data --
      but this is a good reminder to always check with an ablation rather
      than assume. See notebooks/milestone2_toy_fusion.ipynb for that
      ablation run live.
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
