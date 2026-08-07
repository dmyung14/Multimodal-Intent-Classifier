import { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

const MODEL_CHOICES = [
  { id: "T", label: "Text only (M5) — accuracy 0.421 (majority-class baseline 0.224)" },
  { id: "TA", label: "Text + Audio (M5) — accuracy 0.411 (baseline 0.224)" },
  { id: "TV", label: "Text + Video (M5) — accuracy 0.336 (baseline 0.224)" },
  { id: "TAV", label: "Text + Audio + Video, concatenated (M5) — accuracy 0.346 (baseline 0.224)" },
  { id: "AV", label: "Audio + Video, no text (M5) — accuracy 0.196 (baseline 0.224)" },
  { id: "A", label: "Audio only (M5) — accuracy 0.150 — BELOW the 0.224 baseline (worse than always guessing)" },
  { id: "V", label: "Video only (M5) — accuracy 0.112 — BELOW the 0.224 baseline (worse than always guessing)" },
  { id: "MISA", label: "Text + Audio + Video, MISA fusion (M3) — accuracy 0.333 on a DIFFERENT, class-balanced test set (baseline 0.050) — not directly comparable to the M5 numbers above" },
];

const NEEDS_TEXT = new Set(["T", "TA", "TV", "TAV", "MISA"]);
// Combos needing only audio accept a standalone audio file; combos needing
// only video (or needing both, since audio+video are extracted from the
// same clip) accept a video file -- mirrors backend/registry.py's
// MODEL_REQUIREMENTS exactly.
const NEEDS_AUDIO_ONLY = new Set(["A", "TA"]);
const NEEDS_VIDEO_ONLY = new Set(["V", "TV"]);
const NEEDS_BOTH = new Set(["AV", "TAV", "MISA"]);

export default function Home() {
  const [modelChoice, setModelChoice] = useState("T");
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("model_choice", modelChoice);
    if (text) formData.append("text", text);
    if (audioFile) formData.append("audio", audioFile);
    if (videoFile) formData.append("video", videoFile);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json();
      if (!response.ok) {
        setError(body.error || "Request failed.");
      } else {
        setResult(body);
      }
    } catch (err) {
      setError(`Could not reach the backend at ${BACKEND_URL}. Is it running?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <h1>Multimodal Intent Inference</h1>
      <p className="intro">
        Every model here is honestly labeled with its own real, measured
        accuracy — text-only wins in every controlled comparison this
        project has run so far. MISA&apos;s number comes from a different,
        smaller test set with a different baseline than the seven M5
        combinations above it, so it isn&apos;t directly comparable by raw
        accuracy — and audio-only/video-only are both worse than always
        guessing the most common intent, not just weak. This pipeline has
        no speech-to-text: type what was said yourself, even for models
        that also use audio/video.
      </p>

      <form onSubmit={handleSubmit} className="card">
        <div className="field">
          <label className="field-label" htmlFor="model-choice">
            Model
          </label>
          <select
            id="model-choice"
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
          >
            {MODEL_CHOICES.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </div>

        {NEEDS_TEXT.has(modelChoice) && (
          <div className="field">
            <label className="field-label" htmlFor="text-input">
              Text (what was said)
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
            />
          </div>
        )}

        {NEEDS_AUDIO_ONLY.has(modelChoice) && (
          <div className="field">
            <label className="field-label" htmlFor="audio-input">
              Audio file (wav/mp3/m4a/etc.)
            </label>
            <input
              id="audio-input"
              type="file"
              accept="audio/*"
              onChange={(e) => setAudioFile(e.target.files[0] || null)}
            />
          </div>
        )}

        {(NEEDS_VIDEO_ONLY.has(modelChoice) || NEEDS_BOTH.has(modelChoice)) && (
          <div className="field">
            <label className="field-label" htmlFor="video-input">
              Video file (mp4/webm)
            </label>
            <input
              id="video-input"
              type="file"
              accept="video/mp4,video/webm"
              onChange={(e) => setVideoFile(e.target.files[0] || null)}
            />
          </div>
        )}

        <button type="submit" disabled={loading} style={{ marginTop: 6 }}>
          {loading ? "Predicting…" : "Predict intent"}
        </button>
      </form>

      {error && (
        <div className="error" style={{ marginTop: 20 }}>
          <strong>Error</strong>
          {error}
        </div>
      )}

      {result && (
        <div className="card" style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p className="result-value">{result.predicted_intent}</p>

          <div style={{ marginTop: 16 }}>
            <div className="result-row">
              <span className="k">Model</span>
              <span className="v">{result.model_choice}</span>
            </div>
            <div className="result-row">
              <span className="k">Confidence</span>
              <span className="v">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          <p className="caveat">{result.explanation.calibration_caveat}</p>

          {result.explanation.top_words && result.explanation.top_words.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <label className="field-label">Top contributing words</label>
              <ul className="word-list">
                {result.explanation.top_words.map((w) => (
                  <li key={w.word}>
                    {w.word}
                    <span className="weight">{w.weight.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
