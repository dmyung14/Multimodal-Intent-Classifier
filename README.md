# Multimodal Intent Inference

A staged, beginner-friendly project for predicting a speaker's **perceived
communicative intent** (and, later, belief) from short conversational clips
using text, audio, and video.

> This is a learning project. It starts deliberately simple — a text-only
> classifier you can run in seconds — and grows one modality and one idea at a
> time. See `CLAUDE.md` for the full plan and the working agreement with Claude
> Code.

## Quick start (5 minutes, no GPU needed)

```bash
# 1. Create the environment (installs Python + libraries)
conda env create -f environment.yml
conda activate mmi

# 2. Run Milestone 0 — a text-only intent classifier on the bundled data
python src/train_text_only.py
```

You should see the script narrate itself through four steps (load → split →
train → evaluate) and finish with accuracy, macro-F1, a per-class report, and a
confusion matrix.

### About that first score

On the tiny bundled dataset (96 sentences) you'll get a **modest** accuracy,
not 99%. That's expected and it's the point: with very little data a model
can't learn much. It gives you something real to poke at. Good first questions
to ask Claude Code:

- "Why is accuracy low here, and would more data help?"
- "Which intents get confused with each other in the confusion matrix, and why?"
- "Open `results/milestone0_predictions.csv` and help me read the mistakes."

## Next step

When you're ready for Milestone 1, install the heavier libraries and try a real
transformer, then move to the actual MIntRec dataset:

```bash
pip install torch transformers datasets accelerate
python src/train_text_only.py --model transformer
```

## How to use Claude Code here

Open this folder in VS Code, run `claude` in the terminal, and it will read
`CLAUDE.md` automatically. Start each session by telling it which milestone
you're on. Ask it to *explain before it codes* — that's how the ML actually
gets into your head.
