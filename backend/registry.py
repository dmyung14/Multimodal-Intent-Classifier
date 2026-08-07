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

try:
    from expected_results import EXPECTED, TOLERANCE  # noqa: E402
except ImportError:
    from backend.expected_results import EXPECTED, TOLERANCE  # noqa: E402

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


def _verify_bundle_metrics(name, bundle):
    """
    Check a loaded checkpoint's stored accuracy/macro_f1 (written by
    scripts/export_checkpoints.py at save time) against the EXPECTED
    real, already-recorded numbers, within TOLERANCE. Returns an error
    string describing the mismatch, or None if the checkpoint is fine.
    A checkpoint from before this metadata existed (missing keys) is
    treated as a failure too -- it can't be verified, so it can't be
    trusted.
    """
    expected = EXPECTED[name]
    acc = bundle.get("accuracy")
    macro_f1 = bundle.get("macro_f1")
    if acc is None or macro_f1 is None:
        return f"{name}: checkpoint is missing accuracy/macro_f1 metadata (predates verification)"
    acc_ok = abs(acc - expected["accuracy"]) <= TOLERANCE
    f1_ok = abs(macro_f1 - expected["macro_f1"]) <= TOLERANCE
    if not (acc_ok and f1_ok):
        return (
            f"{name}: checkpoint accuracy={acc:.4f} macro_f1={macro_f1:.4f} "
            f"does not match expected accuracy={expected['accuracy']:.4f} "
            f"macro_f1={expected['macro_f1']:.4f} (tolerance={TOLERANCE})"
        )
    return None


def load_registry():
    registry = {}
    missing = []
    failed_verification = []

    for combo in LR_COMBOS:
        path = MODELS_DIR / f"lr_{combo}.joblib"
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        bundle = joblib.load(path)
        error = _verify_bundle_metrics(combo, bundle)
        if error:
            failed_verification.append(error)
            continue
        registry[combo] = bundle

    misa_path = MODELS_DIR / "misa.pt"
    if not misa_path.exists():
        missing.append(str(misa_path.relative_to(PROJECT_ROOT)))
    else:
        bundle = torch.load(misa_path, weights_only=False)
        error = _verify_bundle_metrics("MISA", bundle)
        if error:
            failed_verification.append(error)
        else:
            model = MISAModel(
                text_dim=bundle["text_dim"],
                hidden_dim=bundle["hidden_dim"],
                num_classes=bundle["num_classes"],
            )
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            registry["MISA"] = {"model": model, "tfidf": bundle["tfidf"], "id2label": bundle["id2label"]}

    if missing or failed_verification:
        message = "\n"
        if missing:
            message += "Missing model checkpoint(s):\n" + "\n".join(f"  {m}" for m in missing) + "\n"
        if failed_verification:
            message += "Checkpoint(s) that failed accuracy/macro_f1 verification against expected_results.py:\n"
            message += "\n".join(f"  {m}" for m in failed_verification) + "\n"
        message += "\nRun the export script first:\n    python scripts/export_checkpoints.py\n"
        raise RuntimeError(message)

    return registry
