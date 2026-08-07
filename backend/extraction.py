"""
backend/extraction.py

Live audio/video feature extraction for the app's /predict endpoint.
Mirrors two already-reviewed Milestone 2 scripts exactly, just
parameterized for an arbitrary uploaded file instead of a fixed dataset
row:
  - src/generate_mintrec_multimodal.py's extract_audio/extract_frames
    (ffmpeg invocations: mono 16kHz WAV; 5 evenly-spaced JPEG frames)
  - src/extract_mintrec_embeddings.py's embed_audio/embed_video
    (frozen wav2vec2-base / ResNet18, mean-pooled)

Both encoders are FROZEN here too -- .eval(), no gradients, matching
this project's "prefer frozen encoders" rule.
"""
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

AUDIO_SAMPLE_RATE = 16000
N_FRAMES = 5
AUDIO_CKPT = "facebook/wav2vec2-base"


def load_encoders():
    """Load both frozen encoders once. Call at FastAPI startup, not per request."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CKPT)
    audio_model = Wav2Vec2Model.from_pretrained(AUDIO_CKPT)
    audio_model.eval()

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    video_model = torchvision.models.resnet18(weights=weights)
    video_model.fc = torch.nn.Identity()
    video_model.eval()
    video_transforms = weights.transforms()

    return {
        "audio_feature_extractor": feature_extractor,
        "audio_model": audio_model,
        "video_model": video_model,
        "video_transforms": video_transforms,
    }


def _extract_audio_wav(clip_path, work_dir):
    out_path = Path(work_dir) / "audio.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(clip_path),
                "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1", "-vn",
                "-c:a", "pcm_s16le",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr captured)"
        raise RuntimeError(f"ffmpeg failed while extracting audio: {stderr.strip()}") from exc
    return out_path


def _get_duration_seconds(clip_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(clip_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "(no stderr captured)"
        raise RuntimeError(f"ffprobe failed while reading video duration: {stderr}") from exc

    duration_str = result.stdout.strip()
    if duration_str == "N/A" or not duration_str:
        raise RuntimeError(
            "Could not read this video's duration (ffprobe returned no duration — "
            "common for some .webm recordings). Try re-encoding it to mp4 first, "
            "e.g.: ffmpeg -i input.webm output.mp4"
        )
    return float(duration_str)


def _extract_frames(clip_path, work_dir):
    frame_dir = Path(work_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = _get_duration_seconds(clip_path)
    for i in range(N_FRAMES):
        timestamp = duration * (i + 0.5) / N_FRAMES
        frame_path = frame_dir / f"frame_{i}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(clip_path),
                    "-frames:v", "1", str(frame_path),
                ],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr captured)"
            raise RuntimeError(f"ffmpeg failed while extracting frame {i}: {stderr.strip()}") from exc
    return frame_dir


def _read_wav_as_array(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1, f"expected mono audio, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"expected 16kHz audio, got {w.getframerate()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _embed_audio(wav_path, feature_extractor, model):
    audio = _read_wav_as_array(wav_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.numpy()


def _embed_video(frame_dir, model, transforms):
    frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transforms(image).unsqueeze(0)
            embedding = model(tensor).squeeze(0)
            embeddings.append(embedding.numpy())
    return np.mean(embeddings, axis=0)


def extract_live_features(video_path, encoders, work_dir):
    """
    video_path: Path to an uploaded mp4/webm file.
    encoders: dict returned by load_encoders().
    work_dir: Path to a scratch directory for this request's intermediate
              WAV/JPEG files (caller creates and cleans it up).
    Returns: {"audio": np.ndarray(768,), "video": np.ndarray(512,)}
    """
    wav_path = _extract_audio_wav(video_path, work_dir)
    frame_dir = _extract_frames(video_path, work_dir)
    audio_embedding = _embed_audio(
        wav_path, encoders["audio_feature_extractor"], encoders["audio_model"]
    )
    video_embedding = _embed_video(
        frame_dir, encoders["video_model"], encoders["video_transforms"]
    )
    return {"audio": audio_embedding, "video": video_embedding}
