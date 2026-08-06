# Milestone 5 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematic modality ablation with bootstrap confidence intervals, calibration analysis (reliability diagram + ECE), and abstention (risk-coverage sweep) — implementing this project's `CLAUDE.md` founding principle ("the model should be able to abstain") for the first time — plus an honest, baseline-normalized cross-milestone summary.

**Architecture:** Two scripts (7-way modality sweep with bootstrap CIs; calibration + abstention on the winning combination) reusing Milestone 4's already-downloaded dialogue data (target utterances only, no dialogue context — M4 already established context doesn't help here). A narrated notebook ties both together plus a final cross-milestone comparison table.

**Tech Stack:** Python 3.11, `mmi` conda env. `numpy`/`pandas`/`scikit-learn`/`matplotlib` — all already installed. No new dependencies.

## Global Constraints

- New files only — do not modify any M0-M4 file.
- Reuse `data/dialogue/index.csv` and `data/dialogue/embeddings/*.npz` as-is (from Milestone 4) — no new data acquisition.
- Follow established patterns: module docstrings (goal/how-to-run/terms), `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, `"=" * 70` banners, `SystemExit` with a clear pointer to the missing prerequisite.
- No pytest suite — verification is run-the-script, read the output, check generated files.
- **Uncertainty quantification must be statistically appropriate, not mechanically copied from prior milestones.** `LogisticRegression` is deterministic given a fixed dataset — bootstrap the TEST set for confidence intervals (as the design spec specifies), do not fabricate "multiple training seeds" for a classifier that has no random initialization to average over.
- **`f1_score(..., average="macro")` must always be called with an explicit `labels=` argument** (the full sorted list of intent classes) — this exact bug (implicit label set shifting the metric between runs) was found and fixed in Milestone 4's final review; do not reintroduce it here, including inside the bootstrap resampling loop where a resample could easily miss a rare class.
- Report whichever modality combination actually wins — do not assume `TAV` (all three modalities) wins just because it uses the most information; every prior milestone found combining modalities is not automatically better than a strong single modality.
- Small-sample caveats (107 test rows, 10 confidence bins) must be stated plainly in the notebook, not discovered later by a reviewer.
- The project's `commit` skill is used for every commit, not raw `git commit`.
- Working directly on `master`, no worktree (matches all prior plans).
- Known environment quirks: `conda run -n mmi python -c "<multiline>"` fails on this machine — use a temp `.py` file. Never invoke the `mmi` env's `python.exe` directly — it crashes with `STATUS_STACK_BUFFER_OVERRUN`. Always go through `"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python ...`.

---

## Task 1: Modality ablation with bootstrap confidence intervals

**Files:**
- Create: `src/modality_ablation.py`

**Interfaces:**
- Consumes: `data/dialogue/index.csv` and `data/dialogue/embeddings/*.npz` (from Milestone 4, already exist).
- Produces: `results/modality_ablation.csv` with columns `combination, accuracy, accuracy_ci_low, accuracy_ci_high, macro_f1, macro_f1_ci_low, macro_f1_ci_high` (7 rows, one per combination: `T, A, V, TA, TV, AV, TAV`).
- Produces (for Task 2 to import): `load_data()`, `split_data(df)`, `build_features(combo, train_val, test, tfidf)` — importable functions, same pattern as Milestone 3/4 importing model classes across files.

- [ ] **Step 1: Write `src/modality_ablation.py`**

```python
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
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/modality_ablation.py
```

Expected: STEP 1-3 banners, majority-class baseline line, 7 result lines (one per combination with accuracy/macro-F1 and bootstrap CIs), "Saved full results to results\modality_ablation.csv", a "Best combination by accuracy" line with its baseline multiple. Should finish in a few seconds (7 LogisticRegression fits + 14,000 bootstrap resamples total, all fast).

- [ ] **Step 3: Verify results**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd

results = pd.read_csv('results/modality_ablation.csv')
assert len(results) == 7, f'expected 7 combinations, got {len(results)}'
assert set(results['combination']) == {'T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'}
assert (results['accuracy_ci_low'] <= results['accuracy']).all()
assert (results['accuracy'] <= results['accuracy_ci_high']).all()
assert (results['macro_f1_ci_low'] <= results['macro_f1']).all()
assert (results['macro_f1'] <= results['macro_f1_ci_high']).all()
print('All checks passed.')
print(results.to_string(index=False))
"
```

Expected: `All checks passed.` followed by the 7-row table. (Use a temp `.py` file if the multiline `-c` invocation fails on this machine.)

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/modality_ablation.py` explicitly — not `results/` (gitignored). Suggested message subject: `feat: add modality ablation with bootstrap CIs for Milestone 5`.

---

## Task 2: Calibration and abstention analysis

**Files:**
- Create: `src/calibration_abstention.py`

**Interfaces:**
- Consumes: `results/modality_ablation.csv` (Task 1, to determine the winning combination) and imports `load_data`, `split_data`, `build_features` from `src/modality_ablation.py` (Task 1).
- Produces: `results/calibration_bins.csv` (reliability diagram data, `N_BINS` rows) and `results/risk_coverage.csv` (abstention sweep, 21 rows).

- [ ] **Step 1: Write `src/calibration_abstention.py`**

```python
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
```

- [ ] **Step 2: Run it**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python src/calibration_abstention.py
```

Expected: the "best combination" line, STEP 1-3 banners, predicted-probabilities shape, the small-sample caveat note, the reliability-diagram bin table, an `Expected Calibration Error (ECE): 0.xxx` line, the 21-row risk-coverage table, then confirmation both CSVs were saved.

- [ ] **Step 3: Verify results**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi python -c "
import pandas as pd

bins = pd.read_csv('results/calibration_bins.csv')
assert len(bins) == 10, f'expected 10 bins, got {len(bins)}'
assert (bins['count'].sum() == 107), f'bin counts should sum to 107 test rows, got {bins[\"count\"].sum()}'

rc = pd.read_csv('results/risk_coverage.csv')
assert len(rc) == 21, f'expected 21 threshold rows, got {len(rc)}'
assert rc['coverage'].iloc[0] == 1.0, 'coverage at threshold=0.0 should be 1.0 (answer everything)'
assert rc['coverage'].iloc[-1] <= rc['coverage'].iloc[0], 'coverage should not increase as threshold rises'

print('All checks passed.')
"
```

Expected: `All checks passed.` No assertion on whether accuracy actually improves as coverage decreases — per the design spec, that's a real finding to report honestly either way, not a check to enforce. (Use a temp `.py` file if the multiline `-c` invocation fails on this machine.)

- [ ] **Step 4: Commit**

Use the `commit` skill. Stage `src/calibration_abstention.py` explicitly — not `results/` (gitignored). Suggested message subject: `feat: add calibration and abstention analysis for Milestone 5`.

---

## Task 3: Tutorial notebook + cross-milestone summary

**Files:**
- Create: `notebooks/milestone5_evaluation.ipynb`
- Create (scratchpad only, not committed): a Python builder script filling the notebook's cells, same throwaway pattern used for every prior notebook.

**Interfaces:**
- Consumes: `results/modality_ablation.csv`, `results/calibration_bins.csv`, `results/risk_coverage.csv` (Tasks 1-2), and imports `load_data`/`split_data`/`build_features` from `src/modality_ablation.py` (Task 1) rather than redefining them.

This task's brief is narrative, not a complete code block (matching how Milestone 4's Task 5 — the last notebook task — worked: the authoritative source is the already-reviewed scripts, mirror their logic in cells).

- [ ] **Step 1: Scaffold the notebook from the tutorial template**

```bash
python "C:\Users\dbest\.claude\skills\jupyter-notebook\scripts\new_notebook.py" --kind tutorial --title "Milestone 5 - Evaluation, Ablations, Calibration, Abstention" --out "notebooks/milestone5_evaluation.ipynb"
```

- [ ] **Step 2: Write the cell-filling builder script**

Save to the scratchpad. Mirror `src/modality_ablation.py`'s and `src/calibration_abstention.py`'s logic cell-by-cell (import both modules' functions rather than redefining them — `sys.path.insert` the same way M3/M4's notebooks did). Structure, matching every prior notebook's tutorial shape:

1. **Intro** (audience/prerequisites/learning goals) — explain this milestone is about evaluation *methodology*, not a new architecture; prerequisites: run `src/modality_ablation.py` and `src/calibration_abstention.py` once first.
2. **Outline.**
3. **Setup** (imports, `sys.path.insert`, imports from `modality_ablation`).
4. **Step 1 — the 7-way modality ablation**: run the sweep (or load its saved CSV and also demonstrate re-running one combination live, your choice — the point is the reader sees real numbers with real CIs, not just a static table), plot accuracy with error bars (CIs) for all 7 combinations as a bar chart — a new visualization type for this project. State plainly which combination wins and by how much relative to the majority-class baseline.
5. **Markdown cell explaining bootstrapping** in plain language, contrasting explicitly with M3/M4's multi-seed approach: "M3 and M4 retrained neural networks multiple times because their random initialization/shuffling genuinely changes the result each time. `LogisticRegression` here doesn't have that kind of randomness — refitting it on the same data gives the same answer every time. So instead of asking 'how much does training vary,' bootstrapping asks 'how much would my *estimate* of this fixed model's accuracy vary if I'd gotten a slightly different test set.' Different question, different technique."
6. **Step 2 — calibration**: run/import the reliability-diagram computation, plot it (confidence on x-axis, accuracy on y-axis, scatter or bar per bin, with a diagonal reference line for "perfect calibration"), report ECE, explicitly state the 107-row/10-bin small-sample caveat.
7. **Step 3 — abstention**: plot the risk-coverage curve (or a table if a plot is awkward at this data scale), explain what it shows in plain language, state honestly whether restricting to high-confidence predictions actually improved accuracy on this data or not (real result, not assumed).
8. **Cross-milestone summary**: a markdown + code cell building a comparison table across every milestone's real, already-known results — M0 (0.550, toy 8-class), M1 (0.508, real 20-class text), M2 (0.117, concatenation fusion), M3 (0.290±0.023, MISA, 3-seed), M4 (0.290±0.023, cross-modal + dialogue, 3-seed — note this may coincidentally look similar to M3's number; state plainly if so, don't imply a causal reason without checking), M5 (this milestone's own best modality combination). For EACH row, include that milestone's own majority-class/trivial baseline and the accuracy-as-a-multiple-of-baseline (not just raw accuracy) — this is the direct fix for the exact mistake M4's final review caught (comparing M3 and M4's raw accuracies without normalizing for very different baselines). State clearly in prose that this table is directional (different milestones used different data/scale), not a controlled leaderboard.
9. **Look at the mistakes** (optional, if it fits naturally — e.g. the highest-confidence WRONG predictions, which are the most interesting calibration failures).
10. **Exercises** — e.g. "try a different confidence-bin count (5 instead of 10) for the reliability diagram, does ECE change much given so few test rows?", "which single modality alone beats the majority-class baseline by the largest multiple?".
11. **Pitfalls and extensions** — name the 107-row/single-split limitation explicitly (this milestone's own numbers are also subject to the same small-test-set caution M3/M4 already had to learn), and point to real next steps (leave-one-episode-out CV for a more robust evaluation than one fixed 2-episode test split; temperature/Platt scaling if better calibration is ever needed; scaling to the full 2,213-clip dataset).

- [ ] **Step 3: Run the builder script**

```bash
python <path-to-scratchpad>/build_m5_evaluation_notebook.py
```

- [ ] **Step 4: Execute the notebook top-to-bottom**

```bash
"/c/Users/dbest/anaconda3/Scripts/conda.exe" run -n mmi jupyter nbconvert --to notebook --execute --inplace "notebooks/milestone5_evaluation.ipynb"
```

Expected: successful write, no Python traceback (usual harmless symlink/TCP-kernel warnings expected). Should finish in well under a minute — fitting 7-8 small `LogisticRegression` models plus bootstrap resampling is fast.

- [ ] **Step 5: Verify no errors in the executed notebook**

```bash
python -c "
import json
nb = json.load(open('notebooks/milestone5_evaluation.ipynb', encoding='utf-8'))
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

Use the `commit` skill. Stage `notebooks/milestone5_evaluation.ipynb` only (the builder script stays in the scratchpad). Suggested message subject: `feat: add Milestone 5 evaluation notebook`.

---

## Final check

After Task 3, read back the real ablation/calibration/abstention results and the cross-milestone summary table. This plan does not include a `CLAUDE.md` update step — per this project's established pattern, that's a controller-level step, done directly (not delegated) after this plan's final review is clean, using the real results. Since M5 is the last milestone in `CLAUDE.md`'s planned sequence (M6 is an explicitly optional stretch goal), the controller should also consider whether the overall multi-milestone autonomous run ("cycle and complete M3-M5") is now complete and report back to the user, per their original instruction to stop only once finished.
