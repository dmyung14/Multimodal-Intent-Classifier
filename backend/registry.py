"""
backend/registry.py

Loads every trained model checkpoint this app can serve, once, at
startup. Fails loudly with an exact fix-command pointer if any expected
checkpoint is missing -- same convention every M0-M5 script uses for
missing inputs.
"""
import sys
from pathlib import Path

import joblib
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misa_model import MISAModel  # noqa: E402

# Phase 2: all 7 of M5's modality combinations are now exported/served.
LR_COMBOS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]

MODEL_REQUIREMENTS = {
    "T": {"needs_text": True, "needs_video": False},
    "A": {"needs_text": False, "needs_video": True},
    "V": {"needs_text": False, "needs_video": True},
    "TA": {"needs_text": True, "needs_video": True},
    "TV": {"needs_text": True, "needs_video": True},
    "AV": {"needs_text": False, "needs_video": True},
    "TAV": {"needs_text": True, "needs_video": True},
    "MISA": {"needs_text": True, "needs_video": True},
}


def load_registry():
    registry = {}
    missing = []

    for combo in LR_COMBOS:
        path = MODELS_DIR / f"lr_{combo}.joblib"
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        registry[combo] = joblib.load(path)

    misa_path = MODELS_DIR / "misa.pt"
    if not misa_path.exists():
        missing.append(str(misa_path.relative_to(PROJECT_ROOT)))
    else:
        bundle = torch.load(misa_path, weights_only=False)
        model = MISAModel(
            text_dim=bundle["text_dim"],
            hidden_dim=bundle["hidden_dim"],
            num_classes=bundle["num_classes"],
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        registry["MISA"] = {"model": model, "tfidf": bundle["tfidf"], "id2label": bundle["id2label"]}

    if missing:
        raise RuntimeError(
            "\nMissing model checkpoint(s):\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\n\nRun the export script first:\n    python scripts/export_checkpoints.py\n"
        )

    return registry
