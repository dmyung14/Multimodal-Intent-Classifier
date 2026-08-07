# Multimodal Intent Inference App: Design Spec

Date: 2026-08-06
Status: Approved by user, ready for implementation planning

## Context

M0-M5 built and honestly evaluated a series of intent-classification pipelines on real MIntRec data, but every one of them is a script you run from the command line — nothing is servable. This project builds a real app on top of that work: a user provides text and/or a short video clip, picks which of the project's own trained models to run, and gets back a predicted intent plus an explanation of why.

This app **serves the models already built** (M3's MISA, M5's 7-way modality-combination sweep) — it does not train anything new, and does not attempt to improve on M0-M5's real, already-reported accuracy numbers. It also inherits and must honestly surface those milestones' own findings: text-only is the strongest single approach on this data (M5: 0.421 accuracy), audio/video alone carry near-zero signal, and the models are measurably overconfident (M5: ECE 0.089).

## Model menu

The app exposes **8 model choices**, each labeled with its real, already-measured accuracy so the accuracy trade-off from M5's findings is visible to the app's user, not hidden behind a single "the model" abstraction:

- M5's 7 modality combinations (T, A, V, TA, TV, AV, TAV) — each a `TfidfVectorizer` (text combos only) + `StandardScaler` + `LogisticRegression`, per `src/modality_ablation.py`.
- M3's MISA fusion (text+audio+video, no dialogue context) — `src/misa_model.py` / `src/train_misa.py`.

**M4's cross-modal model is explicitly excluded.** It architecturally expects dialogue context (prior utterances from the same conversation), which a single fresh input from an app user does not have. M4's own final review already validated a "no context" variant (context force-masked, statistically indistinguishable from the full model: 0.265 vs. 0.290), but folding a ninth model in with a different constraint story adds real UI/explanation complexity for a v1 app — deferred, not ruled out for a later version.

## Input handling

- **Text**: a text box, typed by the user directly. This pipeline has never included speech-to-text — MIntRec's own "text" column is human-transcribed, never derived from audio — so for any model combo that uses text, the user types what was said themselves. This is stated plainly in the UI, not left implicit.
- **Audio/video**: a single video file upload (mp4/webm). The backend extracts audio (ffmpeg) and frames (ffmpeg) from it and runs them through the same frozen `wav2vec2-base` / `ResNet18` encoders M2 used, live, per request — reusing `src/extract_mintrec_embeddings.py`'s extraction logic rather than reimplementing it.
- Live mic/webcam recording is out of scope for v1; file upload only.

## Explanation

Every prediction returns three layers (as applicable to the chosen model):

1. **Confidence + calibration framing** (always shown): the model's raw confidence for its top prediction, plus a static caveat derived from M5's calibration analysis — e.g. "this family of model is measurably overconfident (ECE 0.089); treat this confidence number as an upper bound, not a precise probability." This is precomputed once from M5's existing results, not recomputed per request (ECE is a property of the model, not of a single prediction).
2. **Top contributing words** (text-involving M5 combos only — T, TA, TV, TAV): the words in the user's typed text with the largest positive contribution to the predicted class, read directly off the fitted `LogisticRegression`'s `.coef_` for that class × the TF-IDF weight of each word present in the input. Not available for MISA (a neural net with no such direct, honest attribution without adding a new technique like SHAP, which is out of scope) or for audio/video-only combos (no text to attribute to).
3. **LLM-generated narrative explanation**: **out of scope for v1.** Wrapping the result in AI-generated plain English requires a new external dependency (an LLM API key, a network call, per-request cost) that nothing in M0-M5 needed. Deferred to a clearly-scoped follow-up rather than bundled in silently.

## Architecture

Two local processes, no deployment, no auth, no database — this is a personal tool running on the developer's machine, matching this project's scope so far.

```
Browser (Next.js frontend)
   │  POST /predict { text, video_file, model_choice }
   ▼
FastAPI backend (mmi conda env)
   │  1. If model_choice needs audio/video and a file was uploaded:
   │     extract audio/frames (ffmpeg) → wav2vec2/ResNet18 embeddings
   │  2. Build the feature vector the same way that model's training
   │     script did (TF-IDF for text combos, raw embeddings for MISA)
   │  3. Run the loaded checkpoint → prediction + class probabilities
   │  4. Assemble the explanation (confidence/calibration always;
   │     top words if text-involving M5 combo)
   ▼
JSON response → frontend renders result panel
```

### Model persistence (new capability this app requires)

None of M0-M5's scripts save trained weights — every "train" script retrains from scratch each run and only ever saved *results* (predictions, confusion matrices), never a checkpoint. Verified directly: no `torch.save`/`joblib.dump`/`pickle.dump` calls anywhere in `src/*.py`.

**New file: `scripts/export_checkpoints.py`.** A one-time export step, run once ahead of building the backend (and re-run only if a model changes). It imports the existing training functions from `src/modality_ablation.py` and `src/train_misa.py` (no changes to those files) and, after training completes exactly as it already does, adds a save step:
- M5's 7 combos: `joblib.dump` the fitted `TfidfVectorizer` (text combos), `StandardScaler`, and `LogisticRegression` as one bundle per combo.
- M3's MISA: `torch.save(model.state_dict())`.

Output lands in `models/` (already a project directory, git-ignored, previously only holding an unrelated `distilbert_intent` experiment).

### Backend: `backend/app.py` (new)

FastAPI app. Loads all 8 checkpoints into memory once at startup — refuses to start with a clear pointer to `python scripts/export_checkpoints.py` if any are missing (same "fail loud with a fix command" convention as every M0-M5 script). One endpoint: `POST /predict`.

### Frontend: `frontend/` (new)

Next.js/React, styled from the `chatgpt-design` skill's extracted tokens (light theme, OpenAI Sans, 4px spacing grid, `#fcfcfc`/`#ececec`/`#0d0d0d`/`#3a83f7` color tokens, no gradients/blur per the extracted anti-patterns). Layout: model picker (8 options, each showing its real accuracy), text box, file upload, submit button, results panel (predicted intent, confidence bar, calibration caveat, highlighted contributing words when applicable).

## Error handling

- Backend startup fails loudly if any checkpoint is missing, naming the exact export command to run.
- A combo requiring audio/video with no file uploaded (or a text-requiring combo with an empty text box) → `400` naming exactly what's missing.
- ffmpeg/encoder extraction failure on a bad upload → surfaced as a specific error to the caller, never silently swallowed or defaulted.
- Frontend shows a distinct, visible error state if the backend is unreachable or returns an error — never a silent blank result.

## Testing

This is the project's first real application code, not a training script — a step up from M0-M5's "run it, read the output" convention is warranted:
- Backend: FastAPI `TestClient` tests on `/predict` — one per input-validation error case (missing text, missing file when required, bad file), one happy-path test per model family (one M5 combo, MISA) using a small fixture clip.
- Frontend: manual verification in a real browser (start the dev server, exercise the golden path and at least one error case), per this project's global CLAUDE.md UI-testing rule — no automated frontend tests for v1.

## Explicitly out of scope for v1

- M4's cross-modal/dialogue model.
- Live mic/webcam recording (file upload only).
- Speech-to-text / auto-transcription (text is always user-typed).
- LLM-generated narrative explanations.
- Auth, database, deployment/hosting — local-only tool for now.
