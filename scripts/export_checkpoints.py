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
