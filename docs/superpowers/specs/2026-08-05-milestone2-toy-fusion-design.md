# Milestone 2, First Increment: Toy Multimodal Fusion Sanity Check

Date: 2026-08-05
Status: Approved

## Context

`CLAUDE.md` defines Milestone 2 as "add frozen audio + video encoders,
concatenation fusion." The project's established pattern (Milestones 0 and 1)
is to prove a pipeline works on small, fast, zero-risk data before spending
time or bandwidth on anything real:

- M0 proved the text-only load → split → train → evaluate loop on a
  96-row toy dataset (`data/sample_intents.csv`).
- M1 proved the same loop scales to the real MIntRec text data
  (`data/mintrec/all.tsv`, 2,224 rows, 20 classes).

M2 needs the same staged approach, but for audio and video. The real
MIntRec audio/video data lives in an 828MB+ archive plus raw video/audio/
speaker-annotation folders on Google Drive — well over this project's
"confirm before downloading >100MB" rule, and not yet acquired. Real
pretrained encoders (e.g. wav2vec2, ResNet) also mean real downloads.

This spec covers only the **first increment**: proving the fusion
mechanics (load three modalities → extract features from each → concatenate
→ classify → evaluate) work end to end, using a synthetic, zero-download
dataset generated from data already in the repo. Real audio/video data and
real pretrained encoders are explicitly out of scope here and will be their
own follow-up increment(s).

## Goal

Prove that "text + audio + video → concatenated feature vector → classifier"
works end to end, on a dataset small and fast enough to sanity-check in
seconds, with **zero new package installs and zero downloads** — consistent
with `CLAUDE.md`'s rules to confirm before big installs/downloads and to
prefer small, lightweight, explainable steps while prototyping.

## Data: synthetic, generated from what already exists

Reuse the same 96 rows / 8 intents from `data/sample_intents.csv` (same
text, same labels — no new text is invented). For each row, generate:

- **Synthetic "audio"**: a short sine wave whose base frequency is tied to
  the intent (each of the 8 intents gets a distinct base frequency, plus
  random jitter per sample so it's not identical for every row of the same
  intent).
- **Synthetic "video"**: a small RGB pixel array whose average color/
  brightness is tied to the intent, again with jitter.

Signal is deliberately tied to the label (not pure noise) so the resulting
notebook can show something meaningful — fusion doing better than text
alone — rather than only "the code runs, scores are near chance."

Both are saved as real files on disk under `data/toy_multimodal/` (e.g.
`audio/0001.npy`, `video/0001.npy`), generated once by a seeded, deterministic
script. Saving actual files (rather than generating in memory at train time)
mirrors how the real MIntRec `audio_data/`/`video_data/` folders work, so the
"load from files" code written now transfers directly to real data later.

A small index file (`data/toy_multimodal/index.csv`) maps each row to its
text, label, and the audio/video file paths.

## Feature extraction ("frozen encoders," classic-signal style)

No new dependencies. Both extractors are small, hand-written, deterministic
functions — "frozen" in the sense that they're fixed, non-learned feature
extractors, standing in for the pretrained neural encoders that will replace
them once real data is in play.

- **Text**: same `TfidfVectorizer` as M0/M1.
- **Audio**: NumPy FFT-based function returning dominant frequency + signal
  energy (2-3 numbers per sample). No librosa/torchaudio.
- **Video**: NumPy function returning mean R/G/B + brightness variance
  (4 numbers per sample). No OpenCV/Pillow.

Function names make the "this is a placeholder" nature explicit (e.g.
`extract_audio_features`, not `load_wav2vec_encoder`), so a future increment
that swaps in real pretrained encoders doesn't get confused with this one.

## Fusion

Concatenate `[TF-IDF text vector | audio features | video features]` into
one row vector per sample, then feed into the same `LogisticRegression`
classifier used in M0/M1 — literally "concatenation fusion."

## New files

- `src/generate_toy_multimodal.py` — one-time, seeded generator script that
  builds `data/toy_multimodal/` (audio `.npy` files, video `.npy` files,
  `index.csv`) from `data/sample_intents.csv`. Re-running it regenerates the
  same files (deterministic via a fixed random seed), so it's safe to re-run.
- `src/train_multimodal_toy.py` — mirrors `train_text_only.py`'s structure
  (load → split → train → evaluate), but loads all three modalities per
  sample, extracts features from each, fuses via concatenation, and reports
  results the same way M0/M1 do (accuracy, macro-F1, per-class report,
  confusion matrix, predictions CSV).
- `notebooks/milestone2_toy_fusion.ipynb` — narrated notebook following the
  same tutorial pattern as the M0/M1 notebooks, including a comparison table
  of text-only accuracy (from M0) vs. this notebook's fused accuracy.

## Data flow

```
data/sample_intents.csv (existing, 96 rows)
        |
        v
src/generate_toy_multimodal.py  (one-time, seeded)
        |
        v
data/toy_multimodal/
  index.csv            (text, label, audio_path, video_path)
  audio/*.npy           (synthetic waveforms)
  video/*.npy           (synthetic RGB arrays)
        |
        v
src/train_multimodal_toy.py
  load_data()      -> reads index.csv, loads each .npy file
  split_data()     -> same stratified train/val/test split as M0/M1
  extract features -> TF-IDF (text) + FFT stats (audio) + color stats (video)
  fuse             -> np.concatenate per sample
  train            -> LogisticRegression.fit()
  evaluate()       -> accuracy, macro-F1, per-class report, confusion matrix,
                       predictions CSV
        |
        v
notebooks/milestone2_toy_fusion.ipynb (same steps, narrated, with plots
  and a text-only-vs-fused comparison table)
```

## Error handling

This is a small, local, deterministic script over synthetic data — no
network calls, no external services, no untrusted input. The only failure
modes worth guarding are: `generate_toy_multimodal.py` requiring
`data/sample_intents.csv` to exist (it already does), and
`train_multimodal_toy.py` requiring `data/toy_multimodal/` to exist (i.e.
requiring the generator to have been run first) — if missing, fail with a
clear message pointing at `generate_toy_multimodal.py`, matching the
project's existing "no silent error recovery" rule. No other special-casing
needed.

## Testing / validation

- Run `generate_toy_multimodal.py` once; spot-check a couple of generated
  `.npy` files load correctly and that `index.csv` has 96 rows.
- Run `train_multimodal_toy.py` from the terminal — should finish in
  seconds, same order of magnitude as M0.
- Execute `milestone2_toy_fusion.ipynb` top-to-bottom via `nbconvert`
  (established project pattern), confirm no errors, and confirm fused
  accuracy is at or above M0's text-only baseline (0.550) — since audio/video
  were deliberately made label-correlated, fusion should help or at least
  not hurt; if it doesn't, that's worth investigating before moving on.

## Explicitly out of scope for this increment

- Real MIntRec audio/video data (the Google Drive archive) — separate future
  increment, needs its own download-size confirmation.
- Real pretrained frozen encoders (wav2vec2, ResNet, etc.) — swapped in once
  real data is in play, in a later increment.
- Any change to the existing M0/M1 scripts or notebooks.
