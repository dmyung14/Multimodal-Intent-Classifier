# Milestone 4: Cross-Modal Transformer + Dialogue Context

Date: 2026-08-06
Status: Approved (autonomous run — see M3's design spec for the process note; same applies here)

## Context

`CLAUDE.md`'s M4 goal: "cross-modal transformer + dialogue context." This is
two ideas at once:

1. **Cross-modal transformer**: instead of MISA's self-attention over 6
   fixed per-modality representations (M3), let modalities directly attend
   to *each other's* information — the mechanism from Tsai et al. 2019's
   "Multimodal Transformer" (MulT), another architecture this project's
   `CLAUDE.md` names as inspiration (via the `declare-lab` reference repo).
2. **Dialogue context**: every milestone through M3 classified each
   utterance independently. Real conversations carry context — what was
   said right before an utterance often disambiguates its intent. This
   milestone is the first to use that.

## A real data-scope problem this milestone surfaces

M2/M3's 300-clip subset was a **stratified sample** (15 clips per intent
class, drawn from across all 43 episodes) — so no two clips were ever
near each other in the same episode. Dialogue context needs the opposite:
utterances that are at least temporally ordered within one episode. The
M2/M3 dataset cannot be reused for this milestone's core capability — a
genuinely new, differently-selected acquisition is required.

**Verified real episode sizes** (queried directly from the same
`THU-IAR/MIntRec` Hugging Face mirror used since M1/M2): 43 episodes
total, 2,213 clips, median 42 clips/episode, mean 51.5, ranging from 20
(`S05/E02`) to 100 (`S05/E15`, `S05/E09`).

> **Correction added after the final whole-branch review.** An earlier
> version of this spec (and of the M4 notebook) described the 10 selected
> episodes as giving "complete, contiguous" dialogue. **That was factually
> wrong and has been corrected throughout.** Those "episode clip counts"
> are counts of *annotated* clips, not of all clips in the episode.
> MIntRec annotates only **7.6%–10.7%** of each of these episodes'
> clips — e.g. `S05/E20` has 53 annotated rows spanning clip numbers 8 to
> 547. Pooled across all 10 episodes, the median gap between an utterance
> and the annotated utterance before it is **7 clip numbers**, and only
> **60 of 502** such pairs (12%) are literally consecutive clips; a full
> 4-context window spans a median of 39 clip numbers end to end.
>
> So a "dialogue window" in this milestone means **"the 4 most recently
> annotated utterances in the same episode"** — typically several scenes
> apart, not four consecutive conversational turns. Decision #3's context
> mechanism is still implemented as described; what changes is the honest
> description of what that context *is*. `src/train_crossmodal.py` prints
> these sparsity numbers at run time (STEP 1b) and trains an explicit
> **"no dialogue context" ablation** (every context position force-masked
> for training *and* evaluation) to measure whether this kind of context
> helps at all. Measured over 3 seeds: full model accuracy 0.290 ± 0.023
> vs. no-context 0.265 ± 0.031 — a gap smaller than the seed-to-seed
> spread. **Dialogue context as available in this dataset does not show a
> measurable benefit.** That is a real finding about MIntRec's annotation
> density, not a bug in the context mechanism, and it is reported as such
> rather than hidden.

## Decisions (documented, not asked)

**1. Select 10 episodes and take every annotated clip in each, ~512 clips
total (~560MB), not a stratified sample.** Chosen for a mix of season
variety (S04/S05/S06 all represented, matching M1/M2's existing coverage)
and per-episode class diversity (13-18 distinct intents present per
episode, out of 20 total). The clip counts below are **annotated** clips,
which is only ~8-10% of each episode's clips (see the correction above):

```
S04/E16 (57 clips, 14 classes)   S05/E19 (61 clips, 18 classes)   S06/E01 (54 clips, 15 classes)
S04/E04 (44 clips, 15 classes)   S05/E18 (60 clips, 18 classes)   S06/E03 (51 clips, 15 classes)
S04/E01 (42 clips, 17 classes)   S05/E20 (53 clips, 17 classes)   S06/E04 (50 clips, 13 classes)
                                                                    S06/E02 (40 clips, 16 classes)
```

10 episodes (vs. fewer) specifically to leave enough independent groups
for a defensible train/val/test split by *episode* (see decision #4) —
with only 4-6 episodes, a 70/15/15 split would leave almost no room for
val/test to be more than one episode each.

**2. Chronological order within an episode = clip-number order.** The
Hugging Face mirror's `raw_data/{season}/{episode}/{clip}.mp4` files are
identified only by an integer clip number, not a timestamp. This
milestone assumes ascending clip number approximates the episode's actual
story order (consistent with how the dataset was described as built —
sequential segments extracted from a continuous episode). This is an
unverified assumption, stated plainly rather than silently relied on;
if it's wrong, "dialogue context" here means "context from *some*
consistent but not necessarily chronological ordering," which still
tests whether a model can use *any* structured neighbor information —
a smaller, but not zero, claim.

**3. Context window: up to 4 preceding annotated utterances, predict the
current one.** For annotated utterance *i* in an episode, gather annotated
utterances `max(0, i-4)` through `i-1` as context (fewer if `i` is near the
start of the episode), and predict utterance *i*'s intent using both its own
features and that context. This is the standard "use dialogue history to
classify the current turn" setup from conversational emotion/intent
literature (e.g. DialogueRNN-style setups) — not full-episode context, which
would be larger and slower for a first cross-modal-transformer milestone.
See the correction above for what "preceding" actually means at MIntRec's
annotation density.

*Implementation note (corrected after review — the spec originally said a
"learned 'no context' placeholder").* Missing context positions are
**zero-filled and marked in a boolean padding mask** which
`nn.TransformerEncoder`'s `src_key_padding_mask` uses to exclude them from
attention entirely. No learned placeholder embedding is used, and none is
needed: a masked position contributes nothing to attention, so whatever
value sits in its slot is irrelevant. This is simpler than a learned
placeholder and strictly less to get wrong — the code is correct as
written and the spec text was the thing that needed fixing. It also turns
out to be the mechanism the "no dialogue context" ablation reuses (force
every context position to masked), at zero model-code cost.

**4. Split by episode, not by utterance — prevents context leakage.**
If utterance 50's context (utterances 46-49) could land in a different
split than utterance 50 itself, that split would leak information across
train/val/test. Split at the *episode* level instead: 7 episodes train, 1
val, 2 test (seeded, non-stratified — stratifying 10 groups by which of
20 classes they contain isn't practically meaningful; a plain seeded
shuffle is used instead, documented rather than dressed up as more
rigorous than it is).

**5. Cross-modal transformer: text-centric MulT-style cross-attention,
not full 6-directional.** The original MulT paper cross-attends every
modality pair in both directions (6 total cross-attention passes). Given
every prior milestone (M0, M1, M2, M3) found text the strongest single
modality by a wide margin, this milestone simplifies to: audio and video
sequences cross-attend *into* text (`audio→text`, `video→text`), producing
a text representation enriched by audio/video context, then a
self-attention layer runs across the dialogue-window dimension (5
positions: the 4 context utterances + the target), then the target
position's final representation is classified. This is a real
simplification from the paper (documented, not hidden) — the same spirit
as M3's MISA simplifications — chosen because implementing and debugging
6-directional cross-attention correctly is a much larger undertaking than
this milestone's learning goal (cross-modal attention exists and does
something different from MISA's self-attention-over-fixed-reps) requires.

**6. Reuse M2's frozen encoders (`wav2vec2-base`, `ResNet18`) — only the
clip *selection* and the output format change.** No new encoder work: the
same two frozen models, loaded the same way.

*Corrected after review — this decision made two claims that turned out
not to hold, and the spec text is what's wrong here, not the code:*

- **"Reused as a library import, not copy-pasted."** `src/extract_dialogue_embeddings.py`
  in fact **duplicates M2's helper functions with modifications** rather
  than importing them. That was the pragmatic choice once decision #7
  landed: M2's helpers pool audio to a single `(768,)` vector and average
  the 5 frames to a single `(512,)` vector, while M4 needs `(8, 768)` and
  `(5, 512)` sequences. Importing them would have meant either changing
  M2's already-reviewed, working functions (risking M2/M3's reproducibility)
  or adding shape-switching flags to them. Duplicating ~40 lines of glue
  and keeping M2 frozen was the smaller risk. Left as-is deliberately;
  refactoring it now would be a real code change with its own risk for no
  behavioural gain.
- **"Clips already embedded in M2/M3 are skipped, saving a little redundant
  work."** This is **impossible** and no such skipping happens. M4 writes a
  different embedding *format* (sequence-level `(8, 768)` / `(5, 512)`, per
  decision #7) to a different *directory* (`data/dialogue/embeddings/`, not
  `data/mintrec_multimodal/embeddings/`). An M2/M3 embedding cannot satisfy
  an M4 lookup, so there is no cross-milestone reuse to be had. The script
  *is* idempotent within M4 — re-running skips `.npz` files it has already
  written to `data/dialogue/embeddings/` — which is the useful half of the
  original claim.

**7. Sequence-level features, not just pooled ones, for the transformer's
cross-attention to operate over.** MISA operated on M2's single pooled
vector per modality per clip. A cross-modal *transformer* needs a short
sequence to attend over, not one vector — otherwise cross-attention
degenerates to a single weighted combination with nothing sequential
about it. This milestone extracts a **short per-modality sequence per
utterance**: for audio, `wav2vec2-base`'s hidden states subsampled to a
fixed length of 8 timesteps (mean-pool consecutive groups of the full
sequence down to 8, rather than fully pooling to 1); for video, the same
5 frames from M2 (already a length-5 sequence, previously averaged down
to 1 — this milestone keeps all 5 as ResNet18 embeddings without
averaging); for text, each token's TF-IDF-weighted embedding is not
sequential in the same way, so text is represented as a length-1
"sequence" (its single TF-IDF vector projected to the model's hidden
dimension) — text is the cross-attention *target*, not source, per
decision #5, so it doesn't need its own multi-step sequence.

## Architecture

```python
class CrossModalDialogueModel(nn.Module):
    text_proj:  Linear(text_dim, H)   # text_dim = TF-IDF size; applied once per utterance -> (batch, H)
    audio_proj: Linear(768, H)        # wav2vec2 hidden size; applied per-timestep -> (batch, 8, H)
    video_proj: Linear(512, H)        # ResNet18 embedding size; applied per-frame -> (batch, 5, H)

    audio_to_text_attn: nn.MultiheadAttention(H, nhead=4, batch_first=True)  # query=text (batch,1,H), key=value=audio (batch,8,H)
    video_to_text_attn: nn.MultiheadAttention(H, nhead=4, batch_first=True)  # query=text (batch,1,H), key=value=video (batch,5,H)

    dialogue_self_attn: nn.TransformerEncoderLayer(H, nhead=4, batch_first=True)  # over the 5-position dialogue window

    classifier: Linear(H, num_classes)   # applied to the target position's output
```

`H = 128` (hidden dim, matching M3's choice for consistency). Per utterance:
`enriched_text = text_proj(text) + audio_to_text_attn(query=text_proj(text), key/value=audio_proj(audio))[0] + video_to_text_attn(query=text_proj(text), key/value=video_proj(video))[0]`
(each cross-attention call's query has sequence length 1 — one text vector attending over the 8 audio or 5 video positions — so its output is also length 1, i.e. shape `(batch, 1, H)`, matching `text_proj(text)`'s shape for the residual sum.)

Per utterance in the window: encode text/audio/video, cross-attend
audio→text and video→text (concatenate or sum the two enriched text
outputs into one per-utterance representation), giving one `H`-dim vector
per utterance in the 5-position window. Stack the window as a sequence,
run `dialogue_self_attn`, take the target (last) position, classify.

Loss: plain cross-entropy on the target utterance's intent — no MISA-style
auxiliary losses this time (those were specific to the shared/private
decomposition idea, not part of cross-modal attention).

## New files

- `src/generate_dialogue_data.py` — selects the 10 episodes, downloads
  clips not already local (reusing Task 1's pattern from M2's
  `generate_mintrec_multimodal.py`), extracts raw audio/frames, builds
  dialogue windows (context + target per utterance), writes
  `data/dialogue/index.csv` with columns
  `sample_id, target_text, target_intent, season, episode, clip, context_sample_ids (semicolon-joined, oldest-first), audio_path, frame_dir`.
- `src/extract_dialogue_embeddings.py` — like M2's embedding extraction,
  but audio kept as an 8-step sequence (not pooled to 1) and video kept
  as all 5 frame embeddings (not averaged) — saves richer `.npz` files
  per clip than M2/M3 used.
- `src/crossmodal_model.py` — `CrossModalDialogueModel` + smoke test,
  same pattern as M3's `misa_model.py`.
- `src/train_crossmodal.py` — load → split-by-episode → train (early
  stopping on val macro-F1, same as M3) → evaluate → compare against M0
  (text-only), M2 (concatenation), and M3 (MISA).
- `notebooks/milestone4_crossmodal_dialogue.ipynb` — narrated version.

## Error handling

Same "no silent recovery" pattern as every prior milestone: each script
fails clearly (`SystemExit`) if its required upstream artifact is
missing, naming the exact prior command to run.

## Testing / validation

No pytest suite (project convention).
- `generate_dialogue_data.py`: verify `index.csv` row count matches the
  sum of the 10 episodes' clip counts (512, per the table above, minus
  any of the first 4 clips per episode that lack full 4-utterance context
  — those still get a row, just with fewer/padded context positions, not
  dropped), spot-check a context chain resolves to real prior rows in the
  same episode.
- `extract_dialogue_embeddings.py`: spot-check `.npz` shapes (`audio`:
  `(8, 768)`, `video`: `(5, 512)`).
- `crossmodal_model.py`: smoke test — random tensors of the right shapes
  produce correctly-shaped logits and a finite loss.
- `train_crossmodal.py`: run it, confirm training loss decreases, report
  real accuracy/macro-F1 against M0/M2/M3 — honestly, whatever the number
  is, and **always alongside the test split's trivial majority-class
  baseline** (0.224 here, because M4's test split is class-imbalanced,
  unlike M3's perfectly balanced one; comparing raw accuracies across the
  two milestones without that normalization is misleading). Given M3's
  final review specifically flagged the risk of crediting the wrong
  architectural change for an accuracy shift, and given this milestone
  changes *two* things at once (cross-modal attention **and** dialogue
  context), the script trains **three** variants on the identical data
  and split, over several seeds:
  1. full — cross-modal attention + dialogue context
  2. no cross-attention — cross-attention replaced with mean-pooling
  3. no dialogue context — every context position force-masked, for
     training *and* evaluation (a train/test-matched condition, not a
     context-trained model evaluated without context)

  Macro-F1 is computed with an explicit `labels=` argument everywhere so
  its denominator is always all 20 intents, not whatever subset a given
  run happens to predict; the confusion matrix likewise uses the full
  20-class label set, matching M3's effective convention.
- Notebook: execute top-to-bottom, zero errors.

## Explicitly out of scope

- Full 6-directional MulT cross-attention (decision #5).
- More than a 4-utterance context window.
- The remaining ~1,700 clips not in the 10 selected episodes.
- Any change to M0-M3's existing files.
