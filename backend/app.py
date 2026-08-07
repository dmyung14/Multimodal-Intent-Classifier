"""
backend/app.py

FastAPI app serving this project's trained M3/M5 models.

HOW TO RUN
    conda activate mmi
    python scripts/export_checkpoints.py   # once, if models/ is empty
    uvicorn backend.app:app --reload --port 8000
"""
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.registry import load_registry, MODEL_REQUIREMENTS
from backend.extraction import load_encoders, extract_live_features, extract_audio_only, extract_video_only
from backend.inference import predict
from backend.explanation import build_explanation

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["registry"] = load_registry()
    STATE["encoders"] = load_encoders()
    yield


app = FastAPI(title="Multimodal Intent Inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_endpoint(
    model_choice: str = Form(...),
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    if model_choice not in STATE["registry"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown or unavailable model_choice '{model_choice}'. "
                f"Available: {sorted(STATE['registry'])}"
            },
        )

    requirements = MODEL_REQUIREMENTS[model_choice]
    needs_audio = requirements["needs_audio"]
    needs_video = requirements["needs_video"]

    if requirements["needs_text"] and not text:
        return JSONResponse(
            status_code=400,
            content={"error": f"model_choice '{model_choice}' requires text input."},
        )
    if needs_audio and needs_video and video is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"model_choice '{model_choice}' requires a video file upload "
                "(audio and video are both extracted from it)."
            },
        )
    if needs_audio and not needs_video and audio is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"model_choice '{model_choice}' requires an audio file upload."},
        )
    if needs_video and not needs_audio and video is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"model_choice '{model_choice}' requires a video file upload."},
        )

    live_features = None
    if needs_audio or needs_video:
        work_dir = Path(tempfile.mkdtemp(prefix="mmi_predict_"))
        try:
            if needs_audio and needs_video:
                video_path = work_dir / "upload.mp4"
                with open(video_path, "wb") as f:
                    shutil.copyfileobj(video.file, f)
                live_features = extract_live_features(video_path, STATE["encoders"], work_dir)
            elif needs_audio:
                audio_path = work_dir / "upload_audio"
                with open(audio_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
                live_features = {"audio": extract_audio_only(audio_path, STATE["encoders"], work_dir)}
            else:
                video_path = work_dir / "upload.mp4"
                with open(video_path, "wb") as f:
                    shutil.copyfileobj(video.file, f)
                live_features = {"video": extract_video_only(video_path, STATE["encoders"], work_dir)}
        except Exception as exc:
            kind = "audio" if (needs_audio and not needs_video) else "video"
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to process uploaded {kind}: {exc}"},
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    bundle = STATE["registry"][model_choice]
    prediction = predict(model_choice, text, live_features, STATE["registry"])
    explanation = build_explanation(model_choice, text, bundle, prediction)

    return {
        "model_choice": model_choice,
        "predicted_intent": prediction["predicted_intent"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "explanation": explanation,
    }
