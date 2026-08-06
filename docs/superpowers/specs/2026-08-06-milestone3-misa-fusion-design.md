# Milestone 3: MISA Fusion

Date: 2026-08-06
Status: Approved (autonomous run — see note below)

**Note on process:** the user authorized fully autonomous execution of M3-M5
("cycle and complete M3-M5... automate the entire process until finished").
This spec was therefore written without the usual interactive
brainstorming Q&A — decisions that would normally be checked with the user
are made here directly, with reasoning documented so they're auditable
after the fact rather than invisible.

## Context

`CLAUDE.md`'s M3 goal: "reimplement ONE fusion method (MISA or Low-rank)
cleanly." The user specified MISA.

M2's real-data increment ended on a concrete, motivating finding:
concatenation fusion let the dense audio/video embeddings (~400x larger
raw magnitude than sparse text TF-IDF) dominate the classifier even after
`StandardScaler`, crowding out the text signal almost entirely (fused
0.117 vs. text-only 0.383 on the same 300-clip subset). MISA
(Modality-Invariant and -Specific Representations, Hazarika et al. 2020)
is a natural next step specifically because it doesn't rely on raw
concatenation working out — it explicitly decomposes each modality into a
*shared* (modality-invariant) subspace and a *private* (modality-specific)
subspace, trained with auxiliary losses, then fuses those learned
representations rather than raw feature blocks of wildly different scale
and dimensionality.

## What MISA actually is (reference architecture)

For each modality, the original paper:
1. Encodes a modality's raw sequence (BiLSTM/BiGRU over per-timestep
   features) into one vector.
2. Projects that vector into a **shared** subspace (one projection, shared
   weights across all modalities — this is what makes it "modality
   invariant") and a **private** subspace (a separate projection per
   modality — "modality specific").
3. Trains with four losses together: task loss (classification
   cross-entropy), a **similarity loss** pulling the three modalities'
   shared representations together (the paper uses Central Moment
   Discrepancy, CMD), a **difference loss** pushing each modality's
   private representation apart from its own shared representation and
   from other modalities' private representations (orthogonality), and a
   **reconstruction loss** (shared + private should reconstruct the
   original encoded representation).
4. Fuses all six representations (3 shared + 3 private) through a
   self-attention transformer layer, then classifies.

## Adaptation decisions for this project (documented, not asked)

**1. Modality encoders: MLPs, not BiLSTMs.** M2 already reduced each real
clip to one pooled vector per modality (text: TF-IDF, ~1,400-dim; audio:
768-dim `wav2vec2-base` embedding, already mean-pooled over time; video:
512-dim `ResNet18` embedding, already averaged over 5 frames) — there is no
per-timestep sequence left to feed a BiLSTM. Re-extracting sequential
(non-pooled) features would mean redoing Milestone 2's embedding
extraction from scratch with a different output shape — a much larger
undertaking than "reimplement one fusion method," and arguably a
different milestone's worth of work. Decision: replace the BiLSTM stage
with a small MLP encoder per modality (`raw_dim -> hidden_dim`, one
hidden layer, ReLU). This keeps MISA's actual contribution — the
shared/private decomposition and its three auxiliary losses — intact and
is what this milestone is actually about; the sequence-encoder choice is
orthogonal to that and swappable later.

**2. Similarity loss: pairwise L2 distance, not CMD.** CMD (matching
distributions via their first few statistical moments) is the original
paper's choice and is mathematically fiddly to implement and to explain
to an ML beginner. A pairwise L2 distance between the three modalities'
shared representations (`mean over pairs of ||shared_i - shared_j||^2`)
captures the same *goal* — "make the shared representations agree with
each other" — with code a beginner can read line by line. Documented here
as a deliberate simplification from the paper, not an oversight.

**3. Difference loss: cosine-similarity-based orthogonality.** For each
modality, penalize `(private_m . shared_m)^2` (squared dot product after
L2-normalizing both vectors) to push private and shared apart, and
similarly penalize private-vs-private dot products across modality pairs.
This matches the paper's orthogonality-constraint intent using a simpler
formulation than its Frobenius-norm version.

**4. Fusion: one `nn.TransformerEncoderLayer` over the 6 representations.**
Stack `[shared_text, shared_audio, shared_video, private_text,
private_audio, private_video]` as a length-6 "sequence" (each position
`hidden_dim`-wide), run one self-attention layer so each representation
can attend to the others, then mean-pool across the 6 positions and feed
a linear classifier head. This is close to the paper's actual fusion
mechanism and is a deliberate step up from M0-M2's `np.hstack`
concatenation — self-attention fusion is genuinely new material for this
project.

**5. Data scope: reuse M2's existing 300-clip subset, not the full 2,213.**
`data/mintrec_multimodal/index.csv` and its cached embeddings already
exist — zero new download needed. Scaling to the full dataset is a
reasonable future extension (flagged in the "out of scope" section below)
but isn't necessary to demonstrate MISA's actual point (does a
shared/private decomposition handle modality-scale imbalance better than
raw concatenation?), and keeps this milestone's resource footprint
bounded rather than open-ended, which matters given this is now running
unsupervised across three milestones.

**6. Text features: identical TF-IDF as M2**, so the only thing that
changes between M2's result and M3's result is the fusion *method*, not
also the input features — necessary for an honest comparison.

**7. This is real gradient-based training, not frozen-encoder feature
concatenation** — a genuine departure from M0-M2. `CLAUDE.md`'s "prefer
frozen encoders over fine-tuning... while prototyping" was about the
*input* encoders (`wav2vec2`, `ResNet18`), which stay frozen — this
milestone trains a new, small, on-top classifier/fusion network, which is
what "reimplement a fusion method" necessarily means. The network itself
is kept small (hidden dim 128, single-layer MLP encoders) specifically so
training stays fast (expected well under a minute on CPU: ~195 training
rows, a few hundred parameters' worth of small linear layers, one
attention layer) — matches the spirit of "small, lightweight while
prototyping" even though it's no longer literally frozen.

**8. The `val` split gets used for the first time in this project.**
Every prior script computed it and printed its size but never touched it
further ("reserved for tuning"). With real training now happening, this
milestone actually uses `val` for early stopping (track val macro-F1 each
epoch, keep the best-val-macro-F1 checkpoint, stop if val stops improving
for a patience window) — worth calling out explicitly since it's the
first time this idle split becomes load-bearing.

## Architecture

```python
class MISAModel(nn.Module):
    text_encoder:  Linear(text_dim, H) -> ReLU
    audio_encoder: Linear(768, H) -> ReLU
    video_encoder: Linear(512, H) -> ReLU

    shared_proj:  Linear(H, H)                  # SAME weights for all 3 modalities
    private_text_proj, private_audio_proj, private_video_proj: Linear(H, H)  # separate per modality

    decoder: Linear(2*H, H)                      # reconstructs encoded repr from [shared; private]

    fusion_layer: nn.TransformerEncoderLayer(d_model=H, nhead=4, dim_feedforward=2*H)
    classifier:   Linear(H, num_classes)          # applied after mean-pooling the 6 fused positions
```

`H = 128` (hidden dim), `num_classes = 20`.

Forward pass, per sample:
1. `enc_t, enc_a, enc_v = text_encoder(text_feat), audio_encoder(audio_emb), video_encoder(video_emb)`
2. `shared_t, shared_a, shared_v = shared_proj(enc_t), shared_proj(enc_a), shared_proj(enc_v)` (same `shared_proj` weights every time)
3. `private_t, private_a, private_v = private_text_proj(enc_t), private_audio_proj(enc_a), private_video_proj(enc_v)`
4. Reconstruction: `recon_t = decoder(cat(shared_t, private_t))`, etc. — compared against `enc_t/enc_a/enc_v` (the *encoded*, not raw, representation — reconstructing the raw 1,400-dim sparse TF-IDF vector is a much harder and less meaningful target than reconstructing the dense 128-dim encoded one).
5. Stack `[shared_t, shared_a, shared_v, private_t, private_a, private_v]` into a `(6, H)` sequence, run through `fusion_layer`, mean-pool over the 6 positions -> `(H,)`.
6. `logits = classifier(fused)`.

## Losses

```
task_loss  = CrossEntropyLoss(logits, label)
sim_loss   = mean over the 3 pairs of ||shared_i - shared_j||^2   (i,j in {t,a,v}, i != j)
diff_loss  = mean over (private_m . shared_m)^2 for m in {t,a,v}
           + mean over (private_i . private_j)^2 for i,j in {t,a,v}, i != j
           (all vectors L2-normalized before the dot product)
recon_loss = mean squared error between [recon_t,recon_a,recon_v] and [enc_t,enc_a,enc_v] (detached targets)

total_loss = task_loss + 0.5*sim_loss + 0.5*diff_loss + 0.5*recon_loss
```

Loss weights (0.5 each for the three auxiliary losses) are literature-typical small values, not tuned — documented as a starting point, not a claimed optimum, consistent with this being a "clean reimplementation to learn from," not a paper-reproduction exercise chasing SOTA.

## Training

- Optimizer: `Adam(lr=1e-3)`.
- Batch size: 16 (195 training rows -> ~13 batches/epoch).
- Max epochs: 100, with early stopping on validation macro-F1 (patience 15 epochs) — keep the best checkpoint (by val macro-F1) in memory, restore it before final test evaluation.
- Expected wall-clock: well under a minute on CPU (tiny network, ~195x100 forward/backward passes on <2000-dim inputs).

## New files

- `src/misa_model.py` — the `MISAModel` class and the four loss functions, importable and unit-testable in isolation (kept separate from the training script so the architecture itself is easy to read without training-loop noise).
- `src/train_misa.py` — load -> split -> train (with early stopping on val) -> evaluate -> compare against M2's concatenation-fusion result on the identical data/split.
- `notebooks/milestone3_misa_fusion.ipynb` — narrated version, including a training-curve plot (train/val loss and val macro-F1 per epoch — new visualization type for this project, since nothing before this trained iteratively) and a head-to-head comparison against M2's real-data result on the identical 300-clip subset.

## Data flow

```
data/mintrec_multimodal/index.csv + embeddings/*.npz   (existing, from M2 -- no new acquisition)
        |
        v
src/train_misa.py
  load_data()      -> same as train_mintrec_multimodal.py
  split_data()     -> same stratified split (now val is actually used)
  featurize        -> TF-IDF (text, fit on train) + load .npz (audio, video) -- same as M2, unscaled raw inputs (the MLP encoders' first layer effectively learns scale-appropriate weights per modality, which is part of MISA's point: it doesn't need external scaling to handle modality-scale imbalance)
  MISAModel        -> src/misa_model.py
  train loop       -> Adam, early stopping on val macro-F1, patience 15
  evaluate()        -> accuracy, macro-F1, confusion matrix, predictions CSV, comparison vs M2's 0.117
        |
        v
notebooks/milestone3_misa_fusion.ipynb (same steps, narrated, + training curve plot)
```

## Error handling

Same pattern as every prior script: `load_data()` fails clearly (`SystemExit`) if `data/mintrec_multimodal/index.csv` or its embeddings don't exist, pointing at the M2 scripts that produce them.

## Testing / validation

No pytest suite (project convention). Verification:
- `src/misa_model.py`: a quick standalone check (run as `python -c` or a small `if __name__ == "__main__"` smoke test) that a forward pass on random tensors of the right shapes produces logits of shape `(batch, 20)` and that all four loss functions return finite scalars — catches shape bugs before the full training script depends on it.
- `src/train_misa.py`: run it, confirm training loss decreases over the first several epochs (evidence the network is actually learning, not stuck), confirm early stopping fires or max epochs completes, report real accuracy/macro-F1 against M2's 0.117 and text-only's 0.383 — honestly, whatever the number is, not adjusted to hit a target.
- Notebook: execute top-to-bottom, zero errors, training curve renders.

## Explicitly out of scope for this milestone

- The full 2,213-clip dataset (documented decision #5 above) — a reasonable future extension.
- True CMD similarity loss, true BiLSTM sequence encoders, hyperparameter tuning of the loss weights — all documented simplifications (decisions #1-#3), not required for "reimplement cleanly."
- Any change to M0/M1/M2's existing files.
