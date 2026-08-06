# Milestone 2, Increment 2: Real MIntRec Multimodal Fusion

Date: 2026-08-05
Status: Approved

## Context

Milestone 2's first increment (`docs/superpowers/specs/2026-08-05-milestone2-toy-fusion-design.md`)
proved the text+audio+video concatenation-fusion pipeline's mechanics work,
using fully synthetic, zero-download data and hand-written classic-signal
feature extractors standing in for real encoders. It shipped and passed
review, including a fix wave that corrected an inaccurate `StandardScaler`
narrative surfaced by an ablation on the toy data.

This increment replaces both stand-ins with the real thing: real MIntRec
video/audio data, and real frozen pretrained encoders. This is the
increment that actually completes `CLAUDE.md`'s M2 goal ("add frozen audio
+ video encoders, concatenation fusion").

## Data source

The official `thuiar/MIntRec` GitHub repo (MIT license) only has code. The
real dataset's raw video lives in two places:

- Google Drive: an 828MB `MIA-datasets.tar.gz` archive plus separate
  `audio_data/`, `video_data/`, `raw_data/`, `speaker_annotations_data/`
  folders (pre-extracted features in an old pickle format, sizes not fully
  explored — not used here).
- Hugging Face mirror `THU-IAR/MIntRec` (CC-BY-SA-4.0, the same source
  already used for M1's text data at `data/mintrec/all.tsv`): a `raw_data/`
  folder containing the **raw `.mp4` clips themselves** — 2,213 files,
  2.33GB total, organized as `raw_data/{season}/{episode}/{clip}.mp4`
  (verified, e.g. `raw_data/S04/E01/103.mp4`). This maps exactly onto the
  `season`/`episode`/`clip` columns already present in `data/mintrec/all.tsv`.

This increment uses the Hugging Face `raw_data/` clips exclusively — same
license already agreed to for M1, scriptable via `huggingface_hub` instead
of a manual Google Drive folder download, and lets us extract our own
audio/frames and run our own encoders rather than reverse-engineering the
Google Drive archive's pickle feature format (consistent with this
project's "write our own clean code" philosophy from `CLAUDE.md`).

## Staged acquisition

Per this project's established pattern (toy data before real data, small
before large) and the `CLAUDE.md` rule to confirm before any download over
~100MB:

- **Default run: a stratified subset — 5 clips per intent class (100 clips
  total)**, sampled from the 20 classes in `data/mintrec/all.tsv`. Expected
  download size: tens of MB (clip lengths average 2.38s per the dataset's
  published statistics).
- **`--full` flag: all 2,224 clips (2.33GB).** Not run as part of this
  increment's default path — a separate, explicit confirmation when the
  project is ready to scale up, exactly like M1's real-vs-toy staging.

## File scope

New files alongside the existing toy-data files (matching the M0→M1
pattern: `train_text_only.py` was never modified to also handle real data,
`train_mintrec.py` was added instead). `src/generate_toy_multimodal.py` and
`src/train_multimodal_toy.py` are untouched — they remain a permanently
runnable, zero-download, instant sanity check.

### `src/generate_mintrec_multimodal.py`

Downloads selected clips via `huggingface_hub.hf_hub_download(repo_id="THU-IAR/MIntRec", repo_type="dataset", filename=f"raw_data/{season}/{episode}/{clip}.mp4")`,
then uses `ffmpeg` (already present in the `mmi` conda environment per
`environment.yml`) to extract:

- **Audio:** mono, 16kHz WAV. 16kHz specifically because that's wav2vec2's
  expected input rate — resampling happens once here, not repeatedly later.
- **Video:** 5 evenly-spaced JPEG frames per clip. Duration comes from
  `ffprobe`; frames are pulled via 5 timestamped single-frame `ffmpeg`
  extracts spread across that duration.

Writes `data/mintrec_multimodal/index.csv` with columns
`sample_id, text, intent, season, episode, clip, audio_path, frame_dir`
(`sample_id` built as `{season}_{episode}_{clip}`, e.g. `S04_E01_103`, a
stable natural key). Selection is a stratified sample (5 rows per `label`
value) from `data/mintrec/all.tsv` unless `--full` is passed.

### `src/extract_mintrec_embeddings.py`

A new step this increment introduces that the toy version didn't need:
running two neural encoders over every clip is real CPU work, and
re-running it on every training iteration (e.g. while experimenting with
the classifier or scaling) would be wasteful. This script runs the frozen
encoders **once per clip** and caches the result.

- **Audio encoder:** `wav2vec2-base` (via `transformers`, frozen — no
  fine-tuning, no gradient computation). `torch`/`transformers` are already
  installed from the M1 transformer sanity check, so this needs no new
  package, only its own ~360MB weight download (confirmed separately from
  this spec's approval, at execution time). Mean-pool the model's output
  sequence over the time dimension → one 768-dim vector per clip.
- **Video encoder:** `ResNet18` (via `torchvision` — a new, small
  dependency, ~45MB weights), pretrained on ImageNet, frozen, penultimate
  layer's output as the embedding (512-dim) per frame. Average the 5
  frames' embeddings → one 512-dim vector per clip.

Saves one `.npz` per sample under `data/mintrec_multimodal/embeddings/`
with `audio` (768,) and `video` (512,) arrays.

### `src/train_mintrec_multimodal.py`

Same load → split → fuse → train → evaluate shape as
`src/train_multimodal_toy.py`. The only real change from the toy version:
`extract_audio_features`/`extract_video_features` become "load the
precomputed `.npz`" instead of FFT/color-stat math — everything downstream
(TF-IDF text features, `np.hstack` concatenation, `StandardScaler` fit on
train only, `LogisticRegression`, the same accuracy/macro-F1/
classification-report/confusion-matrix evaluation) is structurally
unchanged. This script also re-runs the toy increment's ablation (scaled vs.
unscaled, per-modality) on the real embeddings — the toy result ("scaling
hurts") was a toy-data artifact caused by unusually clean synthetic signal,
not a general finding, so it needs re-checking on real, messier data rather
than assumed either way.

### `notebooks/milestone2_mintrec_multimodal.ipynb`

Narrated version following the same tutorial pattern as the other three
notebooks (`milestone0_text_only.ipynb`, `milestone1_mintrec_text.ipynb`,
`milestone2_toy_fusion.ipynb`).

## Data flow

```
data/mintrec/all.tsv (existing, from M1)
        |
        v  (stratified 5-per-class sample, or --full)
src/generate_mintrec_multimodal.py
        |  huggingface_hub.hf_hub_download (raw_data/{season}/{episode}/{clip}.mp4)
        |  ffmpeg (audio -> 16kHz mono wav, video -> 5 evenly-spaced jpegs)
        v
data/mintrec_multimodal/
  index.csv
  raw_clips/{season}/{episode}/{clip}.mp4
  audio/{sample_id}.wav
  frames/{sample_id}/frame_{0..4}.jpg
        |
        v
src/extract_mintrec_embeddings.py
  wav2vec2-base (frozen) -> mean-pool over time -> 768-dim audio embedding
  ResNet18 (frozen)      -> average over 5 frames -> 512-dim video embedding
        v
data/mintrec_multimodal/embeddings/{sample_id}.npz  (audio, video arrays)
        |
        v
src/train_mintrec_multimodal.py
  load_data()      -> reads index.csv, joins to embeddings
  split_data()     -> same stratified train/val/test split as M0/M1/M2-inc1
  extract features -> TF-IDF (text) + load .npz (audio, video)
  fuse             -> np.hstack, StandardScaler fit on train only
  train            -> LogisticRegression.fit()
  evaluate()       -> accuracy, macro-F1, per-class report, confusion matrix,
                       predictions CSV, and the scaled/unscaled/per-modality
                       ablation
        |
        v
notebooks/milestone2_mintrec_multimodal.ipynb (same steps, narrated)
```

## Data layout

```
data/mintrec_multimodal/
  index.csv
  raw_clips/{season}/{episode}/{clip}.mp4   (downloaded, gitignored)
  audio/{sample_id}.wav                      (ffmpeg output, gitignored)
  frames/{sample_id}/frame_{0..4}.jpg        (ffmpeg output, gitignored)
  embeddings/{sample_id}.npz                 (encoder output, gitignored)
```

All gitignored, consistent with the rest of `data/` (matches the existing
`.gitignore`'s `data/*` pattern — no `.gitignore` change needed).

## New installs

- `torchvision` — small package, needed for `ResNet18`. `torch` and
  `transformers` are already installed (M1's transformer sanity check), so
  `wav2vec2-base` needs no new package, only its own weight download.
- Both the `torchvision` install and the `wav2vec2-base`/`ResNet18` weight
  downloads are confirmed at execution time, not implied by this spec's
  approval alone.

## Error handling

Consistent with this project's "no silent error recovery" rule:

- `generate_mintrec_multimodal.py` fails clearly if a clip download fails
  (network error, missing file) rather than silently skipping it — a
  beginner debugging a partial dataset needs to know immediately.
- `extract_mintrec_embeddings.py` requires `data/mintrec_multimodal/index.csv`
  to exist first, with a clear message pointing at
  `generate_mintrec_multimodal.py` if missing (same pattern as
  `train_multimodal_toy.py`'s existing check for `toy_multimodal/index.csv`).
- `train_mintrec_multimodal.py` requires `data/mintrec_multimodal/embeddings/`
  to exist first, with a clear message pointing at
  `extract_mintrec_embeddings.py` if missing.

## Testing / validation

Same run-and-read-output style as M0/M1/M2-increment-1 — no pytest suite:

- Run the generator on the 100-clip subset; spot-check `index.csv` has 100
  rows, a sample `.wav` file is real mono 16kHz audio (via a quick
  `wave`-module read), a sample frame directory has 5 real JPEG files.
- Run the embedding step; spot-check a sample `.npz` has `audio` shape
  `(768,)` and `video` shape `(512,)`.
- Run the training script; report the real accuracy/macro-F1 against
  Milestone 0's 0.550 text-only baseline — this is a genuinely unknown
  number, not reused from the toy increment's 0.700.
- Execute the notebook top-to-bottom via `nbconvert`, confirm zero errors.
- Update `CLAUDE.md`'s M2 entry to reflect real data + real encoders once
  real numbers are in hand.

## Explicitly out of scope for this increment

- The Google Drive archive and its pre-extracted pickle-format features —
  not used.
- The full 2,224-clip dataset — available via `--full`, but not run as part
  of this increment; a separate confirmation when the project is ready to
  scale up.
- Modifying `src/generate_toy_multimodal.py` or `src/train_multimodal_toy.py`
  — they remain the permanent, zero-download sanity check.
- Fine-tuning either encoder — both `wav2vec2-base` and `ResNet18` stay
  frozen, matching `CLAUDE.md`'s "frozen encoders over fine-tuning" rule.
