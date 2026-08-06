"""
Milestone 4: extract SEQUENCE-level frozen encoder embeddings for dialogue
context / cross-modal attention (not pooled to one vector, like M2/M3).
=================================================================
GOAL OF THIS FILE
    Milestone 2 pooled each clip's audio into ONE 768-dim vector (mean over
    all of wav2vec2's timesteps) and each clip's video into ONE 512-dim
    vector (mean over 5 frames) -- fine for a plain classifier, but a
    CROSS-MODAL TRANSFORMER needs something to attend OVER, not a single
    already-collapsed vector. This file keeps a short sequence per
    modality instead of fully pooling it:
      - audio: wav2vec2's per-timestep outputs, subsampled down to a
        fixed 8 steps (adaptive mean-pooling, handles any input length)
      - video: all 5 frame embeddings, kept separate (not averaged)

    Both encoders stay FROZEN (no fine-tuning), same as Milestone 2.

HOW TO RUN
    conda activate mmi
    python src/generate_dialogue_data.py    # once, if you haven't already
    python src/extract_dialogue_embeddings.py

    Reuses the same wav2vec2-base/ResNet18 weights already downloaded for
    Milestone 2 -- no new download. Safe to re-run: samples that already
    have a cached embedding are skipped.
"""

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "dialogue" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "dialogue" / "embeddings"

AUDIO_CKPT = "facebook/wav2vec2-base"
AUDIO_SEQ_LEN = 8


def load_audio_encoder():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    model.eval()
    return feature_extractor, model


def load_video_encoder():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def read_wav_as_array(wav_path):
    """Read a mono 16-bit PCM WAV file into a float32 numpy array in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def subsample_to_fixed_length(sequence, target_len):
    """Adaptively mean-pool a (T, D) tensor down to exactly (target_len, D),
    regardless of T -- handles T < target_len, T == target_len, T > target_len."""
    x = sequence.transpose(0, 1).unsqueeze(0)  # (1, D, T)
    pooled = F.adaptive_avg_pool1d(x, target_len)  # (1, D, target_len)
    return pooled.squeeze(0).transpose(0, 1)  # (target_len, D)


def embed_audio(wav_path, feature_extractor, model):
    audio = read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        sequence = outputs.last_hidden_state.squeeze(0)  # (time_steps, 768)
        pooled = subsample_to_fixed_length(sequence, AUDIO_SEQ_LEN)  # (8, 768)
    return pooled.numpy()


def embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.stack(embeddings, axis=0)  # (5, 512)


def main():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the dialogue data first:\n"
            "    python src/generate_dialogue_data.py\n"
        )

    print("=" * 70)
    print("EXTRACT DIALOGUE EMBEDDINGS (sequence-level, frozen wav2vec2-base + ResNet18)")
    print("=" * 70)

    df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(df)} rows from {INDEX_PATH.name}")

    print("\nLoading frozen audio encoder (wav2vec2-base)...")
    feature_extractor, audio_model = load_audio_encoder()
    print("Loading frozen video encoder (ResNet18)...")
    video_model, video_transforms = load_video_encoder()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        out_path = EMBEDDINGS_DIR / f"{row.sample_id}.npz"
        if out_path.exists():
            continue
        print(f"[{i}/{len(df)}] {row.sample_id}")
        audio_embedding = embed_audio(PROJECT_ROOT / row.audio_path, feature_extractor, audio_model)
        video_embedding = embed_video(PROJECT_ROOT / row.frame_dir, video_model, video_transforms)
        np.savez(out_path, audio=audio_embedding, video=video_embedding)

    print("\n" + "=" * 70)
    print(f"DONE. Wrote embeddings for {len(df)} samples to")
    print(f"{EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}")
    print("Next: python src/train_crossmodal.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
