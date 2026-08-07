"""
backend/tests/test_predict.py

Integration tests for POST /predict. Prerequisite: checkpoints must
already be exported (python scripts/export_checkpoints.py) -- these
tests load the real app, including its real startup (real encoders,
real checkpoints), not mocks.

HOW TO RUN
    conda activate mmi
    python scripts/export_checkpoints.py   # once, if not already done
    pytest backend/tests/test_predict.py -v
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_CLIP = PROJECT_ROOT / "data" / "dialogue" / "raw_clips" / "S04" / "E04" / "374.mp4"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_missing_model_choice_returns_422(client):
    response = client.post("/predict", data={"text": "hello"})
    assert response.status_code == 422  # FastAPI's own required-field validation


def test_unknown_model_choice_returns_400(client):
    response = client.post("/predict", data={"model_choice": "NOT_A_REAL_MODEL", "text": "hello"})
    assert response.status_code == 400
    assert "Unknown or unavailable" in response.json()["error"]


def test_t_combo_missing_text_returns_400(client):
    response = client.post("/predict", data={"model_choice": "T"})
    assert response.status_code == 400
    assert "requires text" in response.json()["error"]


def test_misa_missing_video_returns_400(client):
    response = client.post(
        "/predict", data={"model_choice": "MISA", "text": "I guess we should get going"}
    )
    assert response.status_code == 400
    assert "requires a video" in response.json()["error"]


def test_t_combo_happy_path(client):
    response = client.post(
        "/predict",
        data={"model_choice": "T", "text": "I guess we should get going now"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "T"
    assert isinstance(body["predicted_intent"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]["calibration_caveat"]
    assert body["explanation"]["top_words"] is not None


def test_misa_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "MISA", "text": "I guess we should get going now"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "MISA"
    assert isinstance(body["predicted_intent"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]["top_words"] is None  # MISA has no word attribution


def test_v_combo_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "V"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "V"
    assert isinstance(body["predicted_intent"], str)
    assert body["explanation"]["top_words"] is None  # no text in this combo


def test_tav_combo_happy_path(client):
    with open(FIXTURE_CLIP, "rb") as f:
        response = client.post(
            "/predict",
            data={"model_choice": "TAV", "text": "I guess we should get going now"},
            files={"video": ("374.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_choice"] == "TAV"
    assert isinstance(body["predicted_intent"], str)
    assert body["explanation"]["top_words"] is not None
