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
class, drawn from across all 43 episodes) — deliberately *not* contiguous
runs of dialogue. Dialogue context needs the opposite: complete,
temporally-ordered runs of utterances within an episode. The M2/M3 dataset
cannot be reused for this milestone's core capability — a genuinely new,
differently-selected acquisition is required.

**Verified real episode sizes** (queried directly from the same
`THU-IAR/MIntRec` Hugging Face mirror used since M1/M2): 43 episodes
total, 2,213 clips, median 42 clips/episode, mean 51.5, ranging from 20
(`S05/E02`) to 100 (`S05/E15`, `S05/E09`).

## Decisions (documented, not asked)

**1. Select 10 complete episodes, ~512 clips total (~560MB), not a
stratified sample.** Chosen for a mix of season variety (S04/S05/S06 all
represented, matching M1/M2's existing coverage) and per-episode class
diversity (13-18 distinct intents present per episode, out of 20 total):

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

**3. Context window: up to 4 preceding utterances, predict the current
one.** For utterance *i* in an episode, gather utterances `max(0, i-4)`
through `i-1` as context (fewer if `i` is near the start of the episode —
padded with a learned "no context" placeholder, masked out of attention),
and predict utterance *i*'s intent using both its own features and that
context. This is the standard "use dialogue history to classify the
current turn" setup from conversational emotion/intent literature (e.g.
DialogueRNN-style setups) — not full-episode context, which would be
larger and slower for a first cross-modal-transformer milestone.

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

**6. Reuse M2's frozen encoders (`wav2vec2-base`, `ResNet18`), same
embedding extraction approach — only the clip *selection* changes.** No
new encoder work; `src/extract_mintrec_embeddings.py`'s logic is reused
(as a library import, not copy-pasted) for the newly selected clips. Any
clip already embedded from M2/M3's run is skipped (idempotent, same as
before) — a handful of the 512 clips may already overlap with the earlier
300-clip sample by chance, saving a little redundant work.

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
  is. Given M3's final review specifically flagged the risk of crediting
  the wrong architectural change for an accuracy shift, this script
  should also report a plain "self-attention only, no cross-modal
  attention" ablation variant alongside the full model, so this
  milestone doesn't repeat that mistake.
- Notebook: execute top-to-bottom, zero errors.

## Explicitly out of scope

- Full 6-directional MulT cross-attention (decision #5).
- More than a 4-utterance context window.
- The remaining ~1,700 clips not in the 10 selected episodes.
- Any change to M0-M3's existing files.
