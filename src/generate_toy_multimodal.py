"""
Milestone 2, increment 1: generate a synthetic, zero-download multimodal
toy dataset from the existing text-only sample.
=================================================================
GOAL OF THIS FILE
    Milestone 2 is about fusing text + audio + video. Before touching any
    real audio/video data (which lives in an 800MB+ archive we haven't
    downloaded yet), we manufacture tiny FAKE audio and video "clips" for
    each of the 96 sentences already in data/sample_intents.csv, so we can
    build and sanity-check the fusion pipeline in seconds, offline.

    The fake audio is a short sine wave; the fake video is a small solid-ish
    color image. Neither is "real" data -- but each one's frequency/color is
    deliberately tied to its intent label (with random jitter), so a
    classifier CAN learn something from them. That lets the next script
    show fusion actually helping, not just "the code runs."

HOW TO RUN
    conda activate mmi
    python src/generate_toy_multimodal.py

    Finishes in under a second and writes files under data/toy_multimodal/.
    Re-running it is safe -- it's seeded, so it always produces the exact
    same files.

TERMS YOU'LL SEE
    - sine wave : a smooth, single-pitch sound wave; the simplest way to
                  represent a tone as numbers (amplitude at each moment
                  in time)
    - RGB       : each pixel of a color image stored as three numbers
                  (Red, Green, Blue), each 0-255
    - seed      : a fixed starting point for "random" numbers, so
                  re-running the same code produces identical output
                  every time
"""

import colorsys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_INTENTS_PATH = PROJECT_ROOT / "data" / "sample_intents.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "toy_multimodal"

SEED = 0
SAMPLE_RATE = 8000       # audio samples per second. Real speech models use
                          # 16000+; 8000 is plenty for a fake tone and keeps
                          # files tiny. src/train_multimodal_toy.py's audio
                          # feature extractor must use this same value.
AUDIO_DURATION_S = 0.5    # each fake clip is half a second long
FRAME_SIZE = 16           # each fake video frame is 16x16 pixels

INTENTS = ["greeting", "thank", "apologize", "agree", "disagree", "complain", "request", "reassure"]
# One base audio frequency per intent (Hz), evenly spaced 100Hz apart so
# they're easy for a classifier (and for us, reading the numbers) to tell apart.
BASE_FREQUENCIES = {intent: 200 + 100 * i for i, intent in enumerate(INTENTS)}


def intent_base_color(intent):
    """Evenly space each intent around the color wheel so colors are distinct."""
    hue = INTENTS.index(intent) / len(INTENTS)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return np.array([r, g, b]) * 255


def make_audio_clip(intent, rng):
    """A short sine wave whose frequency is tied to the intent, plus jitter."""
    freq = BASE_FREQUENCIES[intent] + rng.uniform(-15, 15)
    t = np.linspace(0, AUDIO_DURATION_S, int(SAMPLE_RATE * AUDIO_DURATION_S), endpoint=False)
    amplitude = rng.uniform(0.7, 1.0)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    noise = rng.normal(0, 0.02, size=wave.shape)  # a little noise so it's not a perfect tone
    return (wave + noise).astype(np.float32)


def make_video_frame(intent, rng):
    """A small RGB image whose average color is tied to the intent, plus jitter."""
    base_color = intent_base_color(intent)
    frame = np.tile(base_color, (FRAME_SIZE, FRAME_SIZE, 1)).astype(np.float32)
    noise = rng.normal(0, 12, size=frame.shape)
    frame = np.clip(frame + noise, 0, 255)
    return frame.astype(np.uint8)


def main():
    print("=" * 70)
    print("GENERATE TOY MULTIMODAL DATA")
    print("=" * 70)

    df = pd.read_csv(SAMPLE_INTENTS_PATH)
    print(f"Loaded {len(df)} rows from {SAMPLE_INTENTS_PATH.name}")

    audio_dir = OUTPUT_DIR / "audio"
    video_dir = OUTPUT_DIR / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for i, row in df.iterrows():
        sample_id = f"{i:04d}"
        audio_path = audio_dir / f"{sample_id}.npy"
        video_path = video_dir / f"{sample_id}.npy"

        np.save(audio_path, make_audio_clip(row["intent"], rng))
        np.save(video_path, make_video_frame(row["intent"], rng))

        rows.append(
            {
                "sample_id": sample_id,
                "text": row["text"],
                "intent": row["intent"],
                "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                "video_path": video_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    index_df = pd.DataFrame(rows)
    index_path = OUTPUT_DIR / "index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"Wrote {len(index_df)} audio clips to {audio_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(index_df)} video frames to {video_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote index to {index_path.relative_to(PROJECT_ROOT)}")
    print("\nDone. This is fake data standing in for real audio/video -- see")
    print("docs/superpowers/specs/2026-08-05-milestone2-toy-fusion-design.md")
    print("for why, and src/train_multimodal_toy.py for how it's used.")


if __name__ == "__main__":
    main()
