"""
Milestone 0: A text-only intent classifier.
=================================================================
GOAL OF THIS FILE
    Teach you the *entire* machine-learning loop with ONE modality (text),
    nothing hidden, nothing that needs a GPU or a big download:

        data  ->  features  ->  model  ->  train  ->  evaluate

    Once you understand this file end to end, everything later in the
    project (audio, video, fusion) is "the same loop, more modalities".

HOW TO RUN
    conda activate mmi
    python src/train_text_only.py

    That default run uses a lightweight scikit-learn model (TF-IDF +
    Logistic Regression). It finishes in a couple of seconds on any laptop,
    CPU only. Read the printed output top to bottom -- it narrates itself.

    Later, once you've installed the heavier libraries, you can try a real
    transformer instead:

        python src/train_text_only.py --model transformer

TERMS YOU'LL SEE (ask Claude Code to expand on any of these):
    - sample        : one row of data (here, one sentence + its label)
    - feature       : a number the model actually reads (TF-IDF turns words into numbers)
    - label / class : the answer we want to predict (the intent)
    - train/val/test: we LEARN on train, TUNE on val, and only JUDGE on test
    - accuracy      : fraction of predictions that were exactly right
    - macro-F1      : accuracy's fairer cousin -- averages performance across
                      classes so a rare class can't be ignored
    - confusion matrix : a table of "what got predicted as what"
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

# Where things live, computed relative to THIS file so it runs from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_intents.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data():
    """Read the CSV into a table and show what we're working with."""
    df = pd.read_csv(DATA_PATH)
    print("=" * 70)
    print("STEP 1  LOAD THE DATA")
    print("=" * 70)
    print(f"Loaded {len(df)} samples from {DATA_PATH.name}")
    print(f"Columns: {list(df.columns)}")
    print("\nHow many examples we have per intent (class balance matters!):")
    print(df["intent"].value_counts().to_string())
    print("\nFive example rows:")
    print(df.sample(5, random_state=0).to_string(index=False))
    return df


def split_data(df):
    """
    Split into train / validation / test.

    We NEVER let the model see the test set during training. The test score
    is our honest estimate of how it does on sentences it has never met.
    'stratify' keeps each split's class balance similar to the whole set.
    """
    print("\n" + "=" * 70)
    print("STEP 2  SPLIT INTO TRAIN / VALIDATION / TEST")
    print("=" * 70)

    # First carve off 20% for test.
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["intent"]
    )
    # Then carve a validation slice out of what remains (~15% of the whole).
    train, val = train_test_split(
        train_val, test_size=0.1875, random_state=42, stratify=train_val["intent"]
    )
    print(f"train: {len(train)} samples  (model learns from these)")
    print(f"val:   {len(val)} samples  (we'd tune settings on these)")
    print(f"test:  {len(test)} samples  (final honest judgement)")
    return train, val, test


def train_sklearn(train):
    """
    Build and train the lightweight model.

    A Pipeline chains two steps:
      1) TfidfVectorizer : turns each sentence into a row of numbers, giving
         more weight to words that are distinctive rather than common.
      2) LogisticRegression : a simple, fast classifier that learns which
         word-patterns point to which intent.
    'fit' is the moment learning actually happens.
    """
    print("\n" + "=" * 70)
    print("STEP 3  BUILD AND TRAIN THE MODEL (TF-IDF + Logistic Regression)")
    print("=" * 70)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train["text"], train["intent"])
    n_features = len(model.named_steps["tfidf"].vocabulary_)
    print(f"Training done. The model turned the text into {n_features} features.")
    return model


def train_transformer(train, val):
    """
    OPTIONAL upgrade path (Milestone 1 territory).

    Fine-tunes a small pretrained language model. This needs the heavier
    libraries (torch, transformers, datasets) and will download ~300MB of
    model weights the first time. We keep it behind a flag so the default
    run stays instant. Don't worry if this errors before you've installed
    those libraries -- the message will tell you what's missing.
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
            "Install them when you're ready for Milestone 1:\n"
            "    pip install torch transformers datasets accelerate\n"
            "For now, just run without the flag to use the sklearn model."
        )

    print("\n" + "=" * 70)
    print("STEP 3  FINE-TUNE A SMALL TRANSFORMER (distilbert)")
    print("=" * 70)

    labels = sorted(train["intent"].unique())
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}

    ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def to_dataset(frame):
        ds = Dataset.from_dict(
            {
                "text": frame["text"].tolist(),
                "label": [label2id[x] for x in frame["intent"]],
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
        output_dir=str(PROJECT_ROOT / "models" / "distilbert_intent"),
        per_device_train_batch_size=8,
        num_train_epochs=5,
        logging_steps=5,
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

    # Wrap so evaluate() can call .predict(list_of_texts) uniformly.
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
    Judge the model on the held-out test set and print the numbers that
    actually matter. Accuracy alone can lie when classes are imbalanced,
    so we always look at macro-F1 and the per-class breakdown too.
    """
    print("\n" + "=" * 70)
    print("STEP 4  EVALUATE ON THE TEST SET (sentences the model never saw)")
    print("=" * 70)

    preds = model.predict(test["text"])
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
    out = RESULTS_DIR / f"milestone0_predictions_{model_name}.csv"
    pd.DataFrame({"text": test["text"], "true": truth, "predicted": preds}).to_csv(
        out, index=False
    )
    print(f"\nSaved every test prediction to {out.relative_to(PROJECT_ROOT)}")
    print("Open that file and read the rows the model got wrong -- that's")
    print("where real understanding starts.")


def main():
    parser = argparse.ArgumentParser(description="Milestone 0: text-only intent classifier")
    parser.add_argument(
        "--model",
        choices=["sklearn", "transformer"],
        default="sklearn",
        help="'sklearn' (default, instant, CPU) or 'transformer' (Milestone 1, needs torch)",
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
    print("DONE. You just ran a full ML pipeline. Next questions to ask Claude Code:")
    print("  - 'Walk me through what TfidfVectorizer actually produced.'")
    print("  - 'Why is macro-F1 different from accuracy here?'")
    print("  - 'Which intents get confused with each other, and why might that be?'")
    print("=" * 70)


if __name__ == "__main__":
    main()
