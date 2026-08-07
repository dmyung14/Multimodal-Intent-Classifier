"""
backend/inference.py

Runs a single prediction through whichever model the request asked for.
Two model families, two different forward passes, one shared return shape.
"""
import numpy as np
import torch


def predict(model_id, text, live_features, registry):
    """
    model_id: a key present in `registry` (e.g. "T", "MISA").
    text: str or None.
    live_features: {"audio": np.ndarray(768,), "video": np.ndarray(512,)} or None.
    registry: as returned by backend.registry.load_registry().
    Returns: {"predicted_intent": str, "confidence": float, "probabilities": dict[str, float]}
    """
    if model_id == "MISA":
        return _predict_misa(text, live_features, registry["MISA"])
    return _predict_lr(model_id, text, live_features, registry[model_id])


def _predict_lr(combo, text, live_features, bundle):
    parts = []
    if "T" in combo:
        parts.append(bundle["tfidf"].transform([text]).toarray())
    if "A" in combo:
        parts.append(live_features["audio"].reshape(1, -1))
    if "V" in combo:
        parts.append(live_features["video"].reshape(1, -1))
    x = np.hstack(parts)
    x_scaled = bundle["scaler"].transform(x)

    clf = bundle["clf"]
    probs = clf.predict_proba(x_scaled)[0]
    classes = clf.classes_
    top_idx = int(np.argmax(probs))
    return {
        "predicted_intent": str(classes[top_idx]),
        "confidence": float(probs[top_idx]),
        "probabilities": {str(cls): float(p) for cls, p in zip(classes, probs)},
    }


def _predict_misa(text, live_features, misa_entry):
    model = misa_entry["model"]
    tfidf = misa_entry["tfidf"]
    id2label = misa_entry["id2label"]

    text_feat = torch.from_numpy(tfidf.transform([text]).toarray().astype(np.float32))
    audio_feat = torch.from_numpy(live_features["audio"].astype(np.float32)).unsqueeze(0)
    video_feat = torch.from_numpy(live_features["video"].astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(text_feat, audio_feat, video_feat)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    top_idx = int(np.argmax(probs))
    return {
        "predicted_intent": id2label[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {id2label[i]: float(p) for i, p in enumerate(probs)},
    }
