"""
Milestone 4: acquire real MIntRec dialogue data (complete episodes) and
build context windows for dialogue-aware intent classification.
=================================================================
GOAL OF THIS FILE
    Milestones 2-3 used a STRATIFIED SAMPLE of clips (15 per intent class,
    drawn from across all 43 episodes) -- deliberately not contiguous
    dialogue. That's the wrong shape of data for THIS milestone: dialogue
    context needs complete, temporally-ordered runs of utterances within
    an episode, so a model can look at what was said just before the
    utterance it's trying to classify.

    This file downloads 10 COMPLETE episodes (~512 clips total) instead of
    a stratified sample, and builds a "dialogue window" per utterance: its
    own clip plus up to 4 preceding clips from the same episode, ordered
    by clip number (used here as a proxy for chronological order -- see
    docs/superpowers/specs/2026-08-06-milestone4-crossmodal-dialogue-design.md
    for why, and the caveat that this is an unverified assumption).

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py

    Downloads ~560MB across 10 episodes -- a real download, similar in
    scale to Milestone 2's data acquisition, and will take several
    minutes (512 clips x ~7 ffmpeg/ffprobe calls each). Safe to re-run:
    already-downloaded clips are skipped.

TERMS YOU'LL SEE
    - dialogue window : the current utterance plus a handful of the
                          utterances immediately before it in the same
                          conversation
    - context          : the "before" utterances in a dialogue window --
                          information the model gets to see but isn't
                          being asked to classify
    - target            : the one utterance in a window whose intent the
                          model is actually being asked to predict
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_TSV_PATH = PROJECT_ROOT / "data" / "mintrec" / "all.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dialogue"

REPO_ID = "THU-IAR/MIntRec"
AUDIO_SAMPLE_RATE = 16000
N_FRAMES = 5
CONTEXT_SIZE = 4

# 10 complete episodes selected for a mix of season variety and per-episode
# class diversity (13-18 of the 20 intents present per episode) -- see the
# design spec for the full table and reasoning.
SELECTED_EPISODES = [
    ("S04", "E16"), ("S04", "E04"), ("S04", "E01"),
    ("S05", "E19"), ("S05", "E18"), ("S05", "E20"),
    ("S06", "E01"), ("S06", "E03"), ("S06", "E04"), ("S06", "E02"),
]

# Same known-missing rows as Milestone 2 (verified against the HF mirror's
# file listing) -- none of these fall inside SELECTED_EPISODES, but the
# check is kept for safety/consistency in case episodes are ever changed.
KNOWN_MISSING = {
    ("S05", "E07", 96), ("S05", "E15", 83), ("S05", "E15", 82),
    ("S05", "E07", 82), ("S05", "E15", 86), ("S05", "E09", 91),
    ("S05", "E07", 87), ("S05", "E15", 85), ("S05", "E15", 94),
    ("S05", "E15", 8), ("S05", "E09", 90),
}


def load_episode_rows():
    """Read all.tsv, keep only rows from SELECTED_EPISODES, sorted by
    (season, episode, clip) so clip order within an episode is ascending."""
    df = pd.read_csv(ALL_TSV_PATH, sep="\t")
    is_missing = df.apply(lambda r: (r["season"], r["episode"], r["clip"]) in KNOWN_MISSING, axis=1)
    df = df[~is_missing]
    mask = df.apply(lambda r: (r["season"], r["episode"]) in SELECTED_EPISODES, axis=1)
    selected = df[mask].sort_values(["season", "episode", "clip"]).reset_index(drop=True)
    print(f"Loaded {len(selected)} rows across {len(SELECTED_EPISODES)} episodes.")
    return selected


def build_windows(episode_rows):
    """For each row, gather up to CONTEXT_SIZE preceding sample_ids from
    the SAME episode (already sorted by clip ascending), oldest-first."""
    rows = []
    for (season, episode), group in episode_rows.groupby(["season", "episode"], sort=False):
        group = group.reset_index(drop=True)
        sample_ids = [f"{season}_{episode}_{row.clip}" for row in group.itertuples(index=False)]
        for i, row in enumerate(group.itertuples(index=False)):
            context_ids = sample_ids[max(0, i - CONTEXT_SIZE):i]  # oldest-first, up to 4
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "target_text": row.text,
                    "target_intent": row.label,
                    "season": season,
                    "episode": episode,
                    "clip": row.clip,
                    "context_sample_ids": ";".join(context_ids),
                }
            )
    return pd.DataFrame(rows)


def download_clip(season, episode, clip):
    """Download one raw clip from the Hugging Face mirror, skipping if already local."""
    local_path = OUTPUT_DIR / "raw_clips" / season / episode / f"{clip}.mp4"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"raw_data/{season}/{episode}/{clip}.mp4",
        )
    except Exception as e:
        raise SystemExit(
            f"\nFailed to download raw_data/{season}/{episode}/{clip}.mp4 from {REPO_ID}: {e}\n"
            "This clip was expected to exist -- check your network connection and try again."
        )
    shutil.copy2(cached_path, local_path)
    return local_path


def extract_audio(clip_path, sample_id):
    """ffmpeg: pull the audio track out as mono 16kHz 16-bit PCM WAV."""
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{sample_id}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-vn",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ],
        check=True, capture_output=True,
    )
    return audio_path


def get_duration_seconds(clip_path):
    """ffprobe: read the clip's duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(clip_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_frames(clip_path, sample_id):
    """ffmpeg: pull N_FRAMES evenly-spaced JPEG frames (midpoints of N equal segments)."""
    frame_dir = OUTPUT_DIR / "frames" / sample_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_seconds(clip_path)
    for i in range(N_FRAMES):
        timestamp = duration * (i + 0.5) / N_FRAMES
        frame_path = frame_dir / f"frame_{i}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(clip_path),
                "-frames:v", "1", str(frame_path),
            ],
            check=True, capture_output=True,
        )
    return frame_dir


def main():
    print("=" * 70)
    print("GENERATE DIALOGUE DATA (10 complete episodes)")
    print("=" * 70)

    episode_rows = load_episode_rows()
    windows = build_windows(episode_rows)

    audio_paths, frame_dirs = [], []
    for i, row in enumerate(windows.itertuples(index=False), start=1):
        print(f"[{i}/{len(windows)}] {row.sample_id} ({row.target_intent})")
        clip_path = download_clip(row.season, row.episode, row.clip)
        audio_path = extract_audio(clip_path, row.sample_id)
        frame_dir = extract_frames(clip_path, row.sample_id)
        audio_paths.append(audio_path.relative_to(PROJECT_ROOT).as_posix())
        frame_dirs.append(frame_dir.relative_to(PROJECT_ROOT).as_posix())

    windows["audio_path"] = audio_paths
    windows["frame_dir"] = frame_dirs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "index.csv"
    windows.to_csv(index_path, index=False)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote {len(windows)} rows to {index_path.relative_to(PROJECT_ROOT)}")
    print(f"Episodes: {len(SELECTED_EPISODES)}, context window size: {CONTEXT_SIZE}")
    print("Next: python src/extract_dialogue_embeddings.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
