"""
Milestone 1: Text-only intent classifier on the REAL MIntRec dataset.
=================================================================
GOAL OF THIS FILE
    Same loop as src/train_text_only.py (load -> split -> train -> evaluate),
    but on the real MIntRec text data instead of the tiny bundled sample:
        - 2,224 utterances (vs. 96)
        - 20 fine-grained intent classes (vs. 8 made-up ones)
        - genuinely imbalanced classes (Complain: 286 rows, Joke: 51 rows)

    Source: text-only columns extracted from the official MIntRec dataset
    (thuiar/MIntRec, ACM MM 2022), mirrored on Hugging Face as
    THU-IAR/MIntRec (all.tsv), license CC-BY-SA-4.0. This file only uses the
    text + label columns -- the real dataset's video/audio/raw clips are
    much larger and are NOT downloaded here (see CLAUDE.md M2+).

HOW TO RUN
    conda activate mmi
    python src/train_mintrec.py

    Default run uses the lightweight sklearn model (TF-IDF + Logistic
    Regression) -- same idea as Milestone 0, just on real data. Finishes in
    a few seconds on any laptop, CPU only.

    The transformer path (--model transformer) also works, but on the full
    2,224 rows it takes several minutes, not seconds -- confirm with
    Claude Code before running it, per this project's training-time rule.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mintrec" / "all.tsv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data():
    """Read the tab-separated MIntRec text data and show what we're working with."""
    df = pd.read_csv(DATA_PATH, sep="\t")
    print("=" * 70)
    print("STEP 1  LOAD THE DATA")
    print("=" * 70)
    print(f"Loaded {len(df)} samples from {DATA_PATH.name}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNumber of distinct intent classes: {df['label'].nunique()}")
    print("\nHow many examples we have per intent (class balance matters!):")
    print(df["label"].value_counts().to_string())
    print("\nFive example rows:")
    print(df[["text", "label"]].sample(5, random_state=0).to_string(index=False))
    return df


def split_data(df):
    """
    Split into train / validation / test, same idea as Milestone 0:
    stratified so every intent (even the rarest, ~51 rows) appears in all
    three splits in roughly the same proportion.
    """
    print("\n" + "=" * 70)
    print("STEP 2  SPLIT INTO TRAIN / VALIDATION / TEST")
    print("=" * 70)

    train_val, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=0.1875, random_state=42, stratify=train_val["label"]
    )
    print(f"train: {len(train)} samples  (model learns from these)")
    print(f"val:   {len(val)} samples  (we'd tune settings on these)")
    print(f"test:  {len(test)} samples  (final honest judgement)")
    return train, val, test


def train_sklearn(train):
    """Same TF-IDF + Logistic Regression pipeline as Milestone 0."""
    print("\n" + "=" * 70)
    print("STEP 3  BUILD AND TRAIN THE MODEL (TF-IDF + Logistic Regression)")
    print("=" * 70)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train["text"], train["label"])
    n_features = len(model.named_steps["tfidf"].vocabulary_)
    print(f"Training done. The model turned the text into {n_features} features.")
    return model


def train_transformer(train, val):
    """
    Same DistilBERT fine-tuning as Milestone 0's transformer path, adapted
    for 20 classes. On the full training split this takes several minutes
    on CPU -- confirm before running, don't launch it silently.
    """
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        raise SystemExit(
            "\nThe transformer path needs extra libraries that aren't installed yet.\n"
            f"Missing: {e.name}\n"
            "Install them first:\n"
            "    pip install torch transformers datasets accelerate\n"
        )

    print("\n" + "=" * 70)
    print("STEP 3  FINE-TUNE A SMALL TRANSFORMER (distilbert)")
    print("=" * 70)

    labels = sorted(train["label"].unique())
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}

    ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def to_dataset(frame):
        ds = Dataset.from_dict(
            {
                "text": frame["text"].tolist(),
                "label": [label2id[x] for x in frame["label"]],
            }
        )
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=32),
            batched=True,
        )

    train_ds, val_ds = to_dataset(train), to_dataset(val)

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "distilbert_mintrec"),
        per_device_train_batch_size=8,
        num_train_epochs=5,
        logging_steps=50,
        report_to="none",
    )

    def metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {"macro_f1": f1_score(pred.label_ids, preds, average="macro")}

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, compute_metrics=metrics,
    )
    trainer.train()

    class HFWrapper:
        def predict(self, texts):
            enc = tokenizer(list(texts), truncation=True, padding=True,
                            max_length=32, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            ids = logits.argmax(dim=1).tolist()
            return [id2label[i] for i in ids]

    return HFWrapper()


def evaluate(model, test, model_name="sklearn"):
    """
    Judge the model on the held-out test set. With 20 classes, a full
    text confusion matrix is too wide for a terminal, so we save it to CSV
    for a notebook to plot instead of printing it here.
    """
    print("\n" + "=" * 70)
    print("STEP 4  EVALUATE ON THE TEST SET (sentences the model never saw)")
    print("=" * 70)

    preds = model.predict(test["text"])
    truth = test["label"].tolist()

    acc = accuracy_score(truth, preds)
    macro_f1 = f1_score(truth, preds, average="macro")
    print(f"Accuracy : {acc:.3f}   (share of predictions that were exactly right)")
    print(f"Macro-F1 : {macro_f1:.3f}   (fairer average across all 20 intents)")

    print("\nPer-class report (precision/recall/F1 for each intent):")
    print(classification_report(truth, preds, zero_division=0))

    RESULTS_DIR.mkdir(exist_ok=True)

    labels = sorted(test["label"].unique())
    cm = confusion_matrix(truth, preds, labels=labels)
    cm_path = RESULTS_DIR / f"mintrec_confusion_matrix_{model_name}.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path)
    print(f"Confusion matrix (20x20, too wide for terminal) saved to {cm_path.relative_to(PROJECT_ROOT)}")

    out = RESULTS_DIR / f"mintrec_predictions_{model_name}.csv"
    pd.DataFrame({"text": test["text"], "true": truth, "predicted": preds}).to_csv(
        out, index=False
    )
    print(f"Saved every test prediction to {out.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Milestone 1: text-only intent classifier on real MIntRec data")
    parser.add_argument(
        "--model",
        choices=["sklearn", "transformer"],
        default="sklearn",
        help="'sklearn' (default, seconds, CPU) or 'transformer' (several minutes on full data)",
    )
    args = parser.parse_args()

    df = load_data()
    train, val, test = split_data(df)

    if args.model == "sklearn":
        model = train_sklearn(train)
    else:
        model = train_transformer(train, val)

    evaluate(model, test, model_name=args.model)

    print("\n" + "=" * 70)
    print("DONE. Compare this to Milestone 0's toy-data numbers -- ask Claude Code:")
    print("  - 'Why does accuracy look different on 20 real classes vs 8 made-up ones?'")
    print("  - 'Which of the 20 intents are hardest, and why might that be?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
