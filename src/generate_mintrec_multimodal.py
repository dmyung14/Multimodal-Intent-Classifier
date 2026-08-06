"""
Milestone 2, increment 2: acquire real MIntRec video/audio and extract raw
audio/frame files for the real frozen-encoder pipeline.
=================================================================
GOAL OF THIS FILE
    Milestone 2's toy increment (src/generate_toy_multimodal.py) proved the
    fusion pipeline's mechanics on fully synthetic, invented data. This
    file replaces that synthetic data with the REAL thing: real short
    video clips from the MIntRec dataset (dialogue from the Superstore TV
    show), downloaded from the THU-IAR/MIntRec mirror on Hugging Face, with
    real audio tracks and real video frames extracted from each clip via
    ffmpeg.

    src/generate_toy_multimodal.py and src/train_multimodal_toy.py are left
    untouched -- they remain a permanent, zero-download, instant sanity
    check. This is a new, separate pipeline for the real data.

HOW TO RUN
    conda activate mmi
    python src/generate_mintrec_multimodal.py

    Default run downloads a STRATIFIED SUBSET: 15 clips per intent class
    (300 clips total, ~330MB). This is a real download over this project's
    "confirm before >100MB" rule -- confirm with Claude Code before running
    this the first time.

    python src/generate_mintrec_multimodal.py --full

    Downloads all 2,213 available clips (~2.3GB) instead of the subset --
    a much bigger download, needing its own separate confirmation.

    Safe to re-run: already-downloaded clips are skipped, not re-fetched.

TERMS YOU'LL SEE
    - stratified sample : picking N examples from EACH class, rather than N
                           examples overall, so rare classes aren't left out
    - ffmpeg / ffprobe   : command-line tools for working with audio/video
                           files. ffprobe reads a file's metadata (like its
                           duration); ffmpeg converts or extracts from it.
    - WAV                : an uncompressed audio file format. The real audio
                           encoder used in the next script (wav2vec2) expects
                           a specific sample rate (16kHz) and channel count
                           (mono, 1 channel) -- we convert to that here so
                           nothing needs resampling later.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_TSV_PATH = PROJECT_ROOT / "data" / "mintrec" / "all.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal"

REPO_ID = "THU-IAR/MIntRec"
SAMPLES_PER_CLASS = 15
SEED = 0
AUDIO_SAMPLE_RATE = 16000  # wav2vec2's expected input rate
N_FRAMES = 5

# (season, episode, clip) rows present in all.tsv with NO matching file in
# the THU-IAR/MIntRec Hugging Face mirror's raw_data/ folder -- verified
# directly against the mirror's file listing (2,213 files vs. all.tsv's
# 2,224 rows). Skipped up front so we never attempt a doomed download.
KNOWN_MISSING = {
    ("S05", "E07", 96),
    ("S05", "E15", 83),
    ("S05", "E15", 82),
    ("S05", "E07", 82),
    ("S05", "E15", 86),
    ("S05", "E09", 91),
    ("S05", "E07", 87),
    ("S05", "E15", 85),
    ("S05", "E15", 94),
    ("S05", "E15", 8),
    ("S05", "E09", 90),
}


def load_available_rows():
    """Read all.tsv and drop the 11 rows known to be missing from the mirror."""
    df = pd.read_csv(ALL_TSV_PATH, sep="\t")
    is_missing = df.apply(lambda r: (r["season"], r["episode"], r["clip"]) in KNOWN_MISSING, axis=1)
    available = df[~is_missing].reset_index(drop=True)
    print(f"Loaded {len(df)} rows from all.tsv; {is_missing.sum()} are known-missing from the")
    print(f"Hugging Face mirror and are excluded, leaving {len(available)} available rows.")
    return available


def sample_stratified(df, n_per_class, seed):
    """Take exactly n_per_class rows from every intent, so no class is left out.

    Uses a manual per-group loop rather than `groupby(...).apply(...)`: on
    pandas 3.x, `GroupBy.apply` silently drops the grouping column ("label")
    from its result, which broke downstream code that reads `row.label`.
    """
    parts = [
        group.sample(n=min(n_per_class, len(group)), random_state=seed)
        for _, group in df.groupby("label")
    ]
    return pd.concat(parts, ignore_index=True)


def download_clip(season, episode, clip):
    """Download one raw clip from the Hugging Face mirror, skipping if already local."""
    local_path = OUTPUT_DIR / "raw_clips" / season / episode / f"{clip}.mp4"
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
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
    parser = argparse.ArgumentParser(description="Acquire real MIntRec audio/video and extract raw audio/frames")
    parser.add_argument("--full", action="store_true", help="download all 2,213 available clips instead of a 300-clip subset")
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE REAL MINTREC MULTIMODAL DATA")
    print("=" * 70)

    available = load_available_rows()
    if args.full:
        selected = available
        print(f"\n--full: using all {len(selected)} available rows.")
    else:
        selected = sample_stratified(available, SAMPLES_PER_CLASS, SEED)
        print(f"\nStratified sample: {SAMPLES_PER_CLASS} rows/class -> {len(selected)} rows.")

    rows = []
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        sample_id = f"{row.season}_{row.episode}_{row.clip}"
        print(f"[{i}/{len(selected)}] {sample_id} ({row.label})")

        clip_path = download_clip(row.season, row.episode, row.clip)
        audio_path = extract_audio(clip_path, sample_id)
        frame_dir = extract_frames(clip_path, sample_id)

        rows.append(
            {
                "sample_id": sample_id,
                "text": row.text,
                "intent": row.label,
                "season": row.season,
                "episode": row.episode,
                "clip": row.clip,
                "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                "frame_dir": frame_dir.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    index_df = pd.DataFrame(rows)
    index_path = OUTPUT_DIR / "index.csv"
    index_df.to_csv(index_path, index=False)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote {len(index_df)} rows to {index_path.relative_to(PROJECT_ROOT)}")
    print("Next: python src/extract_mintrec_embeddings.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
