"""
Milestone 2, increment 2: run frozen pretrained encoders over the real
MIntRec audio/frames and cache the resulting embeddings.
=================================================================
GOAL OF THIS FILE
    src/generate_mintrec_multimodal.py downloaded real clips and extracted
    raw audio (WAV) and video frames (JPEG) from them. This file is where
    those raw files actually meet a real pretrained model:

        audio WAV     -> wav2vec2-base (frozen) -> mean-pool over time -> 768 numbers
        5 JPEG frames -> ResNet18 (frozen)       -> average over frames -> 512 numbers

    Both models are FROZEN: we only ever run them forward to get an
    embedding, never train/fine-tune them (matching CLAUDE.md's "prefer
    frozen encoders over fine-tuning" rule).

    Running two neural networks over hundreds of clips is real CPU work --
    running this ONCE and caching the result means src/train_mintrec_multimodal.py
    (which you'll likely re-run many times while experimenting) can stay
    fast, only ever reading these cached .npz files instead of re-running
    the encoders every time.

HOW TO RUN
    conda activate mmi
    python src/generate_mintrec_multimodal.py   # once, if you haven't already
    python src/extract_mintrec_embeddings.py

    First run downloads wav2vec2-base (~360MB) and ResNet18 (~45MB) weights
    -- confirm with Claude Code before running this the first time. After
    that first download, weights are cached by Hugging Face/torch and
    reused on future runs. Safe to re-run: samples that already have a
    cached embedding are skipped.

TERMS YOU'LL SEE
    - embedding        : a fixed-length list of numbers a neural network
                          produces to summarize its input -- the audio
                          encoder's embedding summarizes a whole clip in
                          768 numbers, the video encoder's summarizes one
                          frame in 512 numbers.
    - mean-pooling      : averaging a sequence of vectors (e.g. wav2vec2's
                          per-timestep outputs) down to a single vector.
    - penultimate layer : the layer just before a network's final
                          classification layer -- its output is a general-
                          purpose feature vector, not tied to ImageNet's
                          specific 1000 classes, which is why it's useful
                          as an embedding for a totally different task.
"""

import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "mintrec_multimodal" / "index.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "mintrec_multimodal" / "embeddings"

AUDIO_CKPT = "facebook/wav2vec2-base"  # self-supervised base model, no task-specific
                                        # head -- a general-purpose frozen audio
                                        # embedding, not tied to e.g. speech recognition.


def load_audio_encoder():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    model.eval()  # frozen: we never call .train() or run a backward pass
    return feature_extractor, model


def load_video_encoder():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # drop the final 1000-class layer; what's left
                                     # outputs ResNet18's 512-dim penultimate embedding
    model.eval()
    return model, weights.transforms()


def read_wav_as_array(wav_path):
    """Read a mono 16-bit PCM WAV file into a float32 numpy array in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def embed_audio(wav_path, feature_extractor, model):
    audio = read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (1, time_steps, 768) -- mean-pool over time
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.numpy()


def embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.mean(embeddings, axis=0)


def main():
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"\n{INDEX_PATH.relative_to(PROJECT_ROOT)} not found.\n"
            "Generate the real MIntRec multimodal data first:\n"
            "    python src/generate_mintrec_multimodal.py\n"
        )

    print("=" * 70)
    print("EXTRACT REAL MINTREC EMBEDDINGS (frozen wav2vec2-base + ResNet18)")
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
    print("Next: python src/train_mintrec_multimodal.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
