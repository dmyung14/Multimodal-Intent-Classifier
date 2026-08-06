"""
Milestone 5: systematic modality ablation with bootstrap confidence intervals.
=================================================================
GOAL OF THIS FILE
    Milestones 2-4 each ran SOME modality ablations, but ad hoc -- a
    handful of combinations, picked to answer that milestone's specific
    question. This file runs the complete sweep: all 7 non-empty subsets
    of {text, audio, video} (T, A, V, TA, TV, AV, TAV), using the same
    simple TF-IDF/pooled-embedding + LogisticRegression classifier for
    every combination, so differences between rows are only ever about
    WHICH MODALITIES are included, not which model architecture.

    Reuses Milestone 4's real dialogue data (data/dialogue/), but only
    each row's TARGET utterance features -- Milestone 4's own final review
    already established that dialogue context doesn't measurably help on
    this data, so leaving it out here keeps the modality question clean.

HOW TO RUN
    conda activate mmi
    python src/modality_ablation.py

    Reuses Milestone 4's already-downloaded data and embeddings -- no new
    download. Fitting 7 LogisticRegression models plus 1,000 bootstrap
    resamples per combination is fast (a few seconds total).

TERMS YOU'LL SEE
    - bootstrap resampling : estimate how much a metric (like accuracy)
      would vary if you had gotten a slightly different test set, by
      repeatedly drawing new "fake" test sets (same size, WITH
      replacement, from the real one) and recomputing the metric each
      time. The spread of those recomputed values is a confidence
      interval -- unlike training multiple random seeds (which measures
      how much a model's TRAINING varies), bootstrapping measures how much
      your ESTIMATE of a fixed, already-trained model's accuracy would
      vary with a different sample of test data.
    - confidence interval (CI) : a range that (roughly) tells you how much
      a point estimate (like "73% accuracy") could plausibly move around
      due to having a limited amount of test data, not due to the model
      itself being different
    - majority-class baseline : the accuracy you'd get by always
      predicting the single most common class -- the "did the model learn
      anything at all" floor
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "dialogue" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "dialogue" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 0
N_BOOTSTRAP = 1000

# Same episode assignment as Milestone 4 (src/train_crossmodal.py), so the
# test set here is the identical 107 rows M4 was evaluated on.
TRAIN_EPISODES = [
    ("S04", "E16"), ("S04", "E04"), ("S04", "E01"), ("S05", "E19"),
    ("S05", "E18"), ("S06", "E03"), ("S06", "E04"),
]
VAL_EPISODES = [("S06", "E02")]
TEST_EPISODES = [("S05", "E20"), ("S06", "E01")]

COMBINATIONS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]


def load_data():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the dialogue data first (Milestone 4):\n"
            "    python src/generate_dialogue_data.py\n"
        )
    if not EMBEDDINGS_DIR.exists() or not any(EMBEDDINGS_DIR.glob("*.npz")):
        raise SystemExit(
            f"\nNo embeddings found in {EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}.\n"
            "Extract embeddings first (Milestone 4):\n"
            "    python src/extract_dialogue_embeddings.py\n"
        )
    print("=" * 70)
    print("STEP 1  LOAD DIALOGUE DATA (target utterances only, no context)")
    print("=" * 70)
    df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(df)} samples from {INDEX_PATH.name}")
    return df


def split_data(df):
    print("\n" + "=" * 70)
    print("STEP 2  SPLIT BY EPISODE (same assignment as Milestone 4)")
    print("=" * 70)

    def episode_in(row, episode_set):
        return (row["season"], row["episode"]) in episode_set

    train_val = df[df.apply(lambda r: episode_in(r, TRAIN_EPISODES + VAL_EPISODES), axis=1)].reset_index(drop=True)
    test = df[df.apply(lambda r: episode_in(r, TEST_EPISODES), axis=1)].reset_index(drop=True)

    print(f"train+val: {len(train_val)} samples across {len(TRAIN_EPISODES) + len(VAL_EPISODES)} episodes")
    print("  (combined -- LogisticRegression needs no validation-based early")
    print("   stopping, so all non-test data is used to fit)")
    print(f"test:      {len(test)} samples across {len(TEST_EPISODES)} episodes "
          "(identical rows Milestone 4 evaluated on)")
    return train_val, test


def pooled_audio(df):
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["audio"].mean(axis=0) for sid in df["sample_id"]])


def pooled_video(df):
    return np.array([np.load(EMBEDDINGS_DIR / f"{sid}.npz")["video"].mean(axis=0) for sid in df["sample_id"]])


def build_features(combo, train_val, test, tfidf):
    """Build fused feature matrices for one modality combination (e.g. 'TA')."""
    parts_train, parts_test = [], []

    if "T" in combo:
        parts_train.append(tfidf.fit_transform(train_val["target_text"]).toarray())
        parts_test.append(tfidf.transform(test["target_text"]).toarray())
    if "A" in combo:
        parts_train.append(pooled_audio(train_val))
        parts_test.append(pooled_audio(test))
    if "V" in combo:
        parts_train.append(pooled_video(train_val))
        parts_test.append(pooled_video(test))

    return np.hstack(parts_train), np.hstack(parts_test)


def bootstrap_ci(truth, preds, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """95% bootstrap confidence interval for a metric on a fixed test set."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    truth_arr, preds_arr = np.array(truth), np.array(preds)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(truth_arr[idx], preds_arr[idx]))
    scores = np.array(scores)
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def run_ablation(train_val, test, all_labels):
    print("\n" + "=" * 70)
    print("STEP 3  RUN THE 7-WAY MODALITY ABLATION")
    print("=" * 70)

    majority_class = train_val["target_intent"].mode()[0]
    majority_preds = [majority_class] * len(test)
    majority_acc = accuracy_score(test["target_intent"], majority_preds)
    print(f"Majority-class baseline ({majority_class}): {majority_acc:.3f} accuracy\n")

    results = []
    for combo in COMBINATIONS:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        x_train, x_test = build_features(combo, train_val, test, tfidf)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(x_train_scaled, train_val["target_intent"])
        preds = clf.predict(x_test_scaled)
        truth = test["target_intent"].tolist()

        acc = accuracy_score(truth, preds)
        macro_f1 = f1_score(truth, preds, average="macro", labels=all_labels, zero_division=0)
        acc_lo, acc_hi = bootstrap_ci(truth, preds, accuracy_score)
        f1_lo, f1_hi = bootstrap_ci(
            truth, list(preds),
            lambda t, p: f1_score(t, p, average="macro", labels=all_labels, zero_division=0),
        )

        print(f"{combo:4s}  accuracy={acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]   "
              f"macro-F1={macro_f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")

        results.append(
            {
                "combination": combo, "accuracy": acc, "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
                "macro_f1": macro_f1, "macro_f1_ci_low": f1_lo, "macro_f1_ci_high": f1_hi,
            }
        )

    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "modality_ablation.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved full results to {out.relative_to(PROJECT_ROOT)}")

    best = results_df.loc[results_df["accuracy"].idxmax()]
    print(f"\nBest combination by accuracy: {best['combination']} ({best['accuracy']:.3f})")
    print(f"(vs. majority-class baseline {majority_acc:.3f} -- "
          f"{best['accuracy'] / majority_acc:.2f}x baseline)")

    return results_df, majority_acc, best["combination"]


def main():
    df = load_data()
    train_val, test = split_data(df)
    all_labels = sorted(df["target_intent"].unique())
    results_df, majority_acc, best_combo = run_ablation(train_val, test, all_labels)

    print("\n" + "=" * 70)
    print("DONE. Next: python src/calibration_abstention.py")
    print(f"(will use the '{best_combo}' combination found above)")
    print("=" * 70)


if __name__ == "__main__":
    main()
