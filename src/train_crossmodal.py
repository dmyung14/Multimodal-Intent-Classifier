"""
Milestone 4: train and evaluate the cross-modal dialogue model on real
MIntRec dialogue data.
=================================================================
GOAL OF THIS FILE
    Same load -> split -> train -> evaluate shape as Milestone 3, but:
      - the split is by EPISODE, not by individual utterance (a target
        utterance's context must stay in the same split as the target,
        or the split would leak information across train/val/test)
      - THREE models are trained and evaluated on the identical data and
        split, so this milestone doesn't repeat Milestone 3's mistake of
        crediting an accuracy change to the wrong mechanism:
          1. full         -- cross-modal attention + dialogue context
          2. no-cross-attn -- cross-attention replaced with mean-pooling
          3. no-context    -- cross-attention kept, but every dialogue
                              context position forced to "padding" for
                              BOTH training and evaluation, so the model
                              only ever sees the target utterance
        Variants 2 and 3 isolate this milestone's two headline ideas
        separately, since M4 changed two things at once.

    A note on "same capacity": variants 1 and 2 declare the same number
    of parameters, but they are NOT equal-capacity. In variant 2 the
    `audio_to_text_attn` / `video_to_text_attn` modules are constructed
    yet never called, so they receive no gradient and are inert -- the
    full model has ~1.2x the *effective* trainable parameters. The
    printed summary reports both counts rather than claiming parity.

    Each variant is run over several seeds (see SEEDS) because a 107-row
    test set across 20 classes is small enough that a single seed's gap
    can easily be noise.

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py        # once
    python src/extract_dialogue_embeddings.py    # once
    python src/train_crossmodal.py

    Trains 3 small models per seed on ~365 rows each -- roughly 15-30s
    per model on CPU, so a few minutes total for all seeds.
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
SEEDS = (0, 1, 2)  # seed 0 is the headline run; the others check it isn't a fluke
NUM_CLASSES = 20

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

# The three conditions compared below. Each is (key, kwargs, human label).
VARIANTS = [
    ("full", dict(use_cross_attention=True, force_mask_context=False),
     "full cross-modal + dialogue context"),
    ("no_cross_attn", dict(use_cross_attention=False, force_mask_context=False),
     "ablation A: no cross-attention (mean-pool instead)"),
    ("no_context", dict(use_cross_attention=True, force_mask_context=True),
     "ablation B: no dialogue context (all context masked)"),
]


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


def report_annotation_sparsity(df):
    """
    MIntRec annotates only a small fraction of each episode's clips, so a
    "dialogue window" here is NOT four consecutive conversational turns.
    Measure and print that plainly rather than assuming otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 1b  HOW CONTIGUOUS IS THIS 'DIALOGUE' ACTUALLY?")
    print("=" * 70)
    print(f"{'episode':>9} {'rows':>5} {'clip range':>12} {'coverage':>9} {'median gap':>11}")
    gaps_all = []
    for (season, episode), group in df.groupby(["season", "episode"]):
        clips = sorted(group["clip"].tolist())
        span = clips[-1] - clips[0] + 1
        gaps = np.diff(clips)
        gaps_all.extend(gaps.tolist())
        print(f"{season + '/' + episode:>9} {len(clips):>5} {f'{clips[0]}-{clips[-1]}':>12} "
              f"{len(clips) / span:>8.1%} {np.median(gaps):>11.1f}")
    gaps_all = np.array(gaps_all)
    print(f"\nPooled over all episodes: median gap between an utterance and the")
    print(f"annotated utterance before it = {np.median(gaps_all):.0f} clip numbers "
          f"(mean {gaps_all.mean():.1f}, max {gaps_all.max()}).")
    print(f"Only {(gaps_all == 1).sum()}/{len(gaps_all)} ({(gaps_all == 1).mean():.1%}) of those pairs are "
          f"literally consecutive clips.")
    print("So 'dialogue context' here means 'the 4 most recently ANNOTATED utterances")
    print("in the same episode' -- typically several scenes apart, not consecutive turns.")
    print("Ablation B below tests whether that context helps at all.")


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

    counts = test["target_intent"].value_counts()
    majority_acc = counts.iloc[0] / len(test)
    print(f"\nTest split is NOT class-balanced: '{counts.index[0]}' alone is "
          f"{counts.iloc[0]}/{len(test)} rows.")
    print(f"Majority-class baseline accuracy on this test split = {majority_acc:.3f} "
          f"-- any model must beat this to have learned anything.")
    return train, val, test, majority_acc


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


def mask_all_context(tensors):
    """
    Return a copy of a (text, audio, video, mask, labels) tuple with EVERY
    dialogue-context position forced to "padding" -- i.e. the normal
    padding mask OR'd with "mask all context positions". Only the target
    position (the last one) stays visible.

    This reuses the model's existing `padding_mask` mechanism, so no model
    code changes are needed: the dialogue self-attention layer simply
    ignores every context position, and the classifier sees a target-only
    representation. Used for BOTH training and evaluation of the
    "no dialogue context" variant, so it is a fair train/test match rather
    than a context-trained model tested without context.
    """
    text, audio, video, mask, labels = tensors
    masked = mask.clone()
    masked[:, :WINDOW_SIZE - 1] = True   # every context position -> ignore
    masked[:, WINDOW_SIZE - 1] = False   # target position is never padding
    return text, audio, video, masked, labels


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
    # `labels=` is explicit on purpose: without it, macro-F1's denominator is
    # however many classes happen to show up in this particular run's
    # truth/predictions, which drifts between runs and makes the number not
    # comparable. Fixing it to all 20 classes keeps the metric stable.
    macro_f1 = f1_score(all_labels, all_preds, labels=list(range(NUM_CLASSES)),
                        average="macro", zero_division=0)
    return avg_loss, macro_f1


def count_effective_params(model):
    """
    Parameters that actually receive gradient. In the no-cross-attention
    ablation the cross-attention modules exist but are never called, so
    they are inert -- counting them would overstate that model's capacity.
    """
    declared = sum(p.numel() for p in model.parameters())
    if model.use_cross_attention:
        return declared, declared
    inert = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("audio_to_text_attn") or name.startswith("video_to_text_attn")
    )
    return declared, declared - inert


def train_model(text_dim, train_tensors, val_tensors, use_cross_attention, label,
                seed=SEED, force_mask_context=False, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 3  TRAIN: {label}  (seed {seed})")
        print("=" * 70)
    if force_mask_context:
        train_tensors = mask_all_context(train_tensors)
        val_tensors = mask_all_context(val_tensors)
        if verbose:
            print("force_mask_context=True: every dialogue-context position is masked out")
            print("for training AND validation -- this model only ever sees the target.")

    text_train, audio_train, video_train, mask_train, labels_train = train_tensors
    text_val, audio_val, video_val, mask_val, labels_val = val_tensors

    torch.manual_seed(seed)
    model = CrossModalDialogueModel(
        text_dim=text_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        use_cross_attention=use_cross_attention
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

        if verbose and (epoch <= 5 or epoch % 10 == 0 or improved):
            marker = " *" if improved else ""
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_f1={train_f1:.3f}  "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.3f}{marker}")

        if epochs_without_improvement >= PATIENCE:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break
    else:
        if verbose:
            print(f"\nReached max epochs ({MAX_EPOCHS}) without early stopping.")

    model.load_state_dict(best_state)
    if verbose:
        declared, effective = count_effective_params(model)
        print(f"Restored best checkpoint (val macro-F1={best_val_f1:.3f}).")
        print(f"Parameters: {declared:,} declared, {effective:,} effective "
              f"(receiving gradient).")
        print("NOTE: early stopping picks WHICH checkpoint to keep. It does not stop the")
        print("model from overfitting in the first place -- train_f1 still reaches 1.000")
        print("early on regardless. Stronger regularization would address the cause.")
    return model, best_val_f1


def evaluate(model, test_tensors, id2label, labels_sorted, label,
             force_mask_context=False, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 4  EVALUATE: {label}")
        print("=" * 70)
    if force_mask_context:
        test_tensors = mask_all_context(test_tensors)
    text_test, audio_test, video_test, mask_test, labels_test = test_tensors

    model.eval()
    with torch.no_grad():
        logits = model(text_test, audio_test, video_test, padding_mask=mask_test)
    preds_id = logits.argmax(dim=1).tolist()
    preds = [id2label[p] for p in preds_id]
    truth = [id2label[l] for l in labels_test.tolist()]

    acc = accuracy_score(truth, preds)
    # Explicit `labels=` again -- macro-F1 over all 20 intents, not just the
    # ones this run happened to predict.
    macro_f1 = f1_score(truth, preds, labels=labels_sorted, average="macro", zero_division=0)
    if verbose:
        print(f"Accuracy : {acc:.3f}")
        print(f"Macro-F1 : {macro_f1:.3f}")
        print("\nPer-class report:")
        print(classification_report(truth, preds, zero_division=0))

    return acc, macro_f1, truth, preds


def main():
    df = load_data()
    report_annotation_sparsity(df)
    train, val, test, majority_acc = split_data(df)

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

    # Train every variant on every seed. Seed 0 is the headline run and is
    # printed verbosely; the extra seeds exist so a single-seed gap on a
    # 107-row test set isn't mistaken for a real effect.
    results = {key: {"acc": [], "f1": [], "val_f1": []} for key, _, _ in VARIANTS}
    headline = {}
    for seed in SEEDS:
        verbose = seed == SEEDS[0]
        if not verbose:
            print(f"\n--- extra seed {seed} (quiet) ---")
        for key, kwargs, label in VARIANTS:
            model, best_val_f1 = train_model(
                text_dim, train_tensors, val_tensors, label=label, seed=seed, verbose=verbose, **kwargs
            )
            acc, f1, truth, preds = evaluate(
                model, test_tensors, id2label, labels_sorted, label=label,
                force_mask_context=kwargs["force_mask_context"], verbose=verbose,
            )
            results[key]["acc"].append(acc)
            results[key]["f1"].append(f1)
            results[key]["val_f1"].append(best_val_f1)
            if not verbose:
                print(f"  seed {seed}  {label:<50} acc={acc:.3f} macro-F1={f1:.3f}")
            if seed == SEEDS[0]:
                headline[key] = (acc, f1, truth, preds, model)

    full_acc, full_f1, truth, preds, _ = headline["full"]

    RESULTS_DIR.mkdir(exist_ok=True)
    # Full 20-class label set for the confusion matrix, matching Milestone 3's
    # convention -- not just the classes this particular run predicted, which
    # would silently change the matrix's shape from run to run.
    cm = confusion_matrix(truth, preds, labels=labels_sorted)
    cm_path = RESULTS_DIR / "crossmodal_confusion_matrix.csv"
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(cm_path)

    out = RESULTS_DIR / "crossmodal_predictions.csv"
    pd.DataFrame({"text": test["target_text"].tolist(), "true": truth, "predicted": preds}).to_csv(out, index=False)

    print("\n" + "=" * 70)
    print(f"RESULTS ACROSS {len(SEEDS)} SEEDS {SEEDS} (identical data, split, hyperparameters)")
    print("=" * 70)
    print(f"{'variant':<50} {'acc (mean+-sd)':>18} {'macro-F1 (mean+-sd)':>21}")
    for key, _, label in VARIANTS:
        a, f = np.array(results[key]["acc"]), np.array(results[key]["f1"])
        print(f"{label:<50} {a.mean():>9.3f} +- {a.std():<5.3f} {f.mean():>12.3f} +- {f.std():<5.3f}")
    print(f"{'trivial majority-class baseline':<50} {majority_acc:>9.3f} {'':<5}       (n/a)")

    full_a = np.array(results["full"]["acc"])
    ctx_a = np.array(results["no_context"]["acc"])
    xattn_a = np.array(results["no_cross_attn"]["acc"])
    print(f"\nCross-attention's contribution (full - no-cross-attn): "
          f"{full_a.mean() - xattn_a.mean():+.3f} accuracy")
    print(f"Dialogue context's contribution (full - no-context):    "
          f"{full_a.mean() - ctx_a.mean():+.3f} accuracy")
    print("Read the second number carefully: given MIntRec's ~8-10% annotation coverage")
    print("per episode (STEP 1b), 'context' here is several scenes back, so a small or")
    print("absent gap is the expected, honest result -- not a bug.")

    print("\n" + "=" * 70)
    print("DONE. Compare against every prior milestone on this project's real data:")
    print(f"  Full cross-modal + dialogue context (M4, seed {SEEDS[0]}): accuracy {full_acc:.3f}, macro-F1 {full_f1:.3f}")
    print(f"  M4 trivial majority-class baseline:                 accuracy {majority_acc:.3f}")
    print("  MISA (M3, 300-clip stratified sample):              accuracy 0.333, macro-F1 0.295")
    print("  M3 trivial majority-class baseline (balanced 60-row test set): accuracy 0.050")
    print("  Concatenation fusion (M2, same sample):             accuracy 0.117, macro-F1 0.096")
    print("Note: M4 uses a DIFFERENT 512-clip dialogue dataset (10 episodes) than M2/M3's")
    print("300-clip stratified sample, and M4's test split is class-imbalanced while M3's")
    print("was perfectly balanced -- so compare each number to ITS OWN baseline, not to")
    print("the other milestone's raw accuracy. The three-variant comparison above, on the")
    print("identical data/split, is the only apples-to-apples experiment here.")
    print(f"Saved predictions to {out.relative_to(PROJECT_ROOT)}")
    print(f"Saved confusion matrix to {cm_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
