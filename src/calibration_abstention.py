"""
Milestone 5: calibration and abstention analysis for the best modality
combination found by the ablation sweep.
=================================================================
GOAL OF THIS FILE
    This project's CLAUDE.md states a founding principle no milestone has
    implemented yet: "Keep outputs honest about uncertainty; the model
    should be able to abstain." This file is where that finally happens,
    for the best-performing classifier from src/modality_ablation.py.

    Two related but different questions:
      - CALIBRATION: when the model says "I'm 80% confident," is it
        actually right about 80% of the time? (Not "is it accurate" --
        a model can be accurate but badly calibrated, or vice versa.)
      - ABSTENTION: if the model is allowed to say "I don't know" on
        low-confidence predictions instead of guessing, how much does
        accuracy improve on the predictions it DOES make?

HOW TO RUN
    conda activate mmi
    python src/modality_ablation.py       # once, if you haven't already
    python src/calibration_abstention.py

    Fast -- refits one LogisticRegression and does some binning/sorting.

TERMS YOU'LL SEE
    - confidence : the model's own predicted probability for whichever
      class it picked (the highest value out of predict_proba's output)
    - reliability diagram : a plot with confidence on one axis and actual
      accuracy on the other, grouped into bins -- a perfectly calibrated
      model's points sit on the diagonal (its confidence always matches
      its real accuracy); points below the diagonal mean the model is
      OVERconfident in that confidence range
    - Expected Calibration Error (ECE) : one number summarizing how far
      off the diagonal the reliability diagram is, on average
    - coverage : the fraction of test samples the model is willing to
      answer (i.e. not abstain on) at a given confidence threshold
    - risk : the error rate among the samples the model DID answer
      (i.e. 1 - accuracy on the covered subset)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from modality_ablation import load_data, split_data, build_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ABLATION_RESULTS_PATH = RESULTS_DIR / "modality_ablation.csv"

SEED = 0
N_BINS = 10


def load_best_combination():
    if not ABLATION_RESULTS_PATH.exists():
        raise SystemExit(
            f"\n{ABLATION_RESULTS_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Run the modality ablation first:\n"
            "    python src/modality_ablation.py\n"
        )
    results = pd.read_csv(ABLATION_RESULTS_PATH)
    best_row = results.loc[results["accuracy"].idxmax()]
    print(f"Best combination from the ablation sweep: {best_row['combination']} "
          f"(accuracy {best_row['accuracy']:.3f})")
    return best_row["combination"]


def fit_best_model(combo, train_val, test):
    """Refit the winning combination's classifier (same recipe as modality_ablation.py)."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train, x_test = build_features(combo, train_val, test, tfidf)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(x_train_scaled, train_val["target_intent"])
    return clf, x_test_scaled


def reliability_diagram_data(truth, pred_probs, class_labels, n_bins=N_BINS):
    """
    truth: true labels (list of strings)
    pred_probs: (n_samples, n_classes) predict_proba output
    class_labels: clf.classes_ -- which column of pred_probs is which class
    Returns (bin_df, confidences, correct): bin_df has one row per
    confidence bin (mean confidence, accuracy, count); confidences/correct
    are per-sample arrays reused by the abstention sweep below.
    """
    confidences = pred_probs.max(axis=1)
    predicted_class_idx = pred_probs.argmax(axis=1)
    predicted_classes = [class_labels[i] for i in predicted_class_idx]
    correct = np.array([p == t for p, t in zip(predicted_classes, truth)])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0, "mean_confidence": np.nan, "accuracy": np.nan})
            continue
        rows.append(
            {
                "bin_low": lo, "bin_high": hi, "count": count,
                "mean_confidence": confidences[in_bin].mean(),
                "accuracy": correct[in_bin].mean(),
            }
        )
    return pd.DataFrame(rows), confidences, correct


def expected_calibration_error(bin_df):
    valid = bin_df.dropna(subset=["accuracy"])
    total = valid["count"].sum()
    return float((valid["count"] / total * (valid["mean_confidence"] - valid["accuracy"]).abs()).sum())


def risk_coverage_sweep(confidences, correct, n_thresholds=21):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    for t in thresholds:
        covered = confidences >= t
        coverage = float(covered.mean())
        if covered.sum() == 0:
            rows.append({"threshold": t, "coverage": 0.0, "accuracy": np.nan, "risk": np.nan})
            continue
        acc = float(correct[covered].mean())
        rows.append({"threshold": t, "coverage": coverage, "accuracy": acc, "risk": 1 - acc})
    return pd.DataFrame(rows)


def main():
    df = load_data()
    train_val, test = split_data(df)
    combo = load_best_combination()

    print("\n" + "=" * 70)
    print(f"STEP 1  REFIT THE BEST COMBINATION ({combo}) AND GET PREDICTED PROBABILITIES")
    print("=" * 70)
    clf, x_test_scaled = fit_best_model(combo, train_val, test)
    pred_probs = clf.predict_proba(x_test_scaled)
    truth = test["target_intent"].tolist()
    print(f"Predicted probabilities shape: {pred_probs.shape}")

    print("\n" + "=" * 70)
    print("STEP 2  CALIBRATION: reliability diagram + Expected Calibration Error")
    print("=" * 70)
    print("Note: only 107 test rows across up to 10 confidence bins -- some bins")
    print("will have very few points. This is a first, coarse pass, not a")
    print("statistically powerful calibration study.")
    bin_df, confidences, correct = reliability_diagram_data(truth, pred_probs, clf.classes_)
    print(bin_df.to_string(index=False))
    ece = expected_calibration_error(bin_df)
    print(f"\nExpected Calibration Error (ECE): {ece:.3f}")

    print("\n" + "=" * 70)
    print("STEP 3  ABSTENTION: risk-coverage sweep")
    print("=" * 70)
    rc_df = risk_coverage_sweep(confidences, correct)
    print(rc_df.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    bin_out = RESULTS_DIR / "calibration_bins.csv"
    rc_out = RESULTS_DIR / "risk_coverage.csv"
    bin_df.to_csv(bin_out, index=False)
    rc_df.to_csv(rc_out, index=False)

    print("\n" + "=" * 70)
    print(f"DONE. ECE={ece:.3f}. Saved calibration bins to {bin_out.relative_to(PROJECT_ROOT)}")
    print(f"and risk-coverage sweep to {rc_out.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
