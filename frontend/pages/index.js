import { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

// Phase 1: only these 2 models are exported/served. Task 9 (Phase 2)
// widens this to all 8 -- see
// docs/superpowers/specs/2026-08-06-multimodal-intent-app-design.md.
const MODEL_CHOICES = [
  { id: "T", label: "Text-only (M5, accuracy 0.421)" },
  { id: "MISA", label: "MISA text+audio+video fusion (M3, accuracy 0.333)" },
];

export default function Home() {
  const [modelChoice, setModelChoice] = useState("T");
  const [text, setText] = useState("");
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
    <main style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Multimodal Intent Inference</h1>
      <p>
        This pipeline has no speech-to-text: type what was said yourself,
        even for models that also use audio/video.
      </p>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 12 }}>
          <label>
            Model:{" "}
            <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
              {MODEL_CHOICES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Text (what was said):
            <br />
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              style={{ width: "100%" }}
            />
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Video file (mp4/webm):
            <br />
            <input
              type="file"
              accept="video/mp4,video/webm"
              onChange={(e) => setVideoFile(e.target.files[0] || null)}
            />
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict intent"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 20, color: "#b00020" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p>
            <strong>Predicted intent:</strong> {result.predicted_intent}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
          </p>
          <p style={{ color: "#666" }}>{result.explanation.calibration_caveat}</p>
          {result.explanation.top_words && (
            <div>
              <strong>Top contributing words:</strong>
              <ul>
                {result.explanation.top_words.map((w) => (
                  <li key={w.word}>
                    {w.word} ({w.weight.toFixed(3)})
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
