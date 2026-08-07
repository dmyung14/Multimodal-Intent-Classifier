"""
backend/explanation.py

Every prediction gets two honest, evidence-based explanation layers
(per docs/superpowers/specs/2026-08-06-multimodal-intent-app-design.md):

1. A calibration caveat -- a STATIC fact about the model family, not
   recomputed per request (calibration is a property of the model, not
   of any single prediction). Only "T" has an actual measured ECE
   (Milestone 5); every other model/combo is honestly labeled as
   "not measured" rather than reusing T's specific number for a model
   that was never calibration-tested.
2. Top contributing words -- ONLY for M5's text-involving
   LogisticRegression combos (T, TA, TV, TAV). Computed as each present
   word's real contribution to the predicted class's linear score:
   coefficient x SCALED feature value (the StandardScaler's transform is
   applied before the score is computed during training, so leaving it
   out here would misrepresent the model's actual math). None for MISA
   (a neural net -- no such direct, honest attribution without adding a
   new technique like SHAP, which is out of scope) and for non-text
   combos (A, V, AV -- no text to attribute to).
"""
import numpy as np

CALIBRATION_CAVEATS = {
    "T": (
        "This model was measured (Milestone 5) to be systematically "
        "overconfident: its average stated confidence (0.485) is higher "
        "than its actual accuracy (0.421) on held-out test data "
        "(Expected Calibration Error = 0.089). Treat the confidence "
        "number above as an upper bound, not a precise probability."
    ),
}
DEFAULT_CAVEAT = (
    "This model's calibration (whether its confidence numbers match its "
    "real accuracy) has not been measured. Treat the confidence number "
    "above as a rough signal only, not a calibrated probability."
)


def top_contributing_words(model_id, text, bundle, prediction):
    if model_id == "MISA" or "T" not in model_id:
        return None

    tfidf = bundle["tfidf"]
    scaler = bundle["scaler"]
    clf = bundle["clf"]

    vocab = tfidf.get_feature_names_out()
    n_text_features = len(vocab)

    class_idx = list(clf.classes_).index(prediction["predicted_intent"])
    coefs = clf.coef_[class_idx][:n_text_features]
    means = scaler.mean_[:n_text_features]
    scales = scaler.scale_[:n_text_features]

    tfidf_vec = tfidf.transform([text]).toarray()[0]
    scaled_vec = (tfidf_vec - means) / scales

    present_idx = np.nonzero(tfidf_vec)[0]
    contributions = [
        {"word": vocab[i], "weight": float(coefs[i] * scaled_vec[i])}
        for i in present_idx
    ]
    contributions.sort(key=lambda c: c["weight"], reverse=True)
    return contributions[:10]


def build_explanation(model_id, text, bundle, prediction):
    return {
        "calibration_caveat": CALIBRATION_CAVEATS.get(model_id, DEFAULT_CAVEAT),
        "top_words": top_contributing_words(model_id, text, bundle, prediction),
    }
