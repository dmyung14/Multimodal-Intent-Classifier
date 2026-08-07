import { useState } from "react";

const BACKEND_URL = "http://localhost:8000";

const MODEL_CHOICES = [
  { id: "T", label: "Text only (M5) — accuracy 0.421" },
  { id: "TA", label: "Text + Audio (M5) — accuracy 0.411" },
  { id: "TV", label: "Text + Video (M5) — accuracy 0.336" },
  { id: "TAV", label: "Text + Audio + Video, concatenated (M5) — accuracy 0.346" },
  { id: "MISA", label: "Text + Audio + Video, MISA fusion (M3) — accuracy 0.333" },
  { id: "AV", label: "Audio + Video, no text (M5) — accuracy 0.196" },
  { id: "A", label: "Audio only (M5) — accuracy 0.150" },
  { id: "V", label: "Video only (M5) — accuracy 0.112" },
];

const NEEDS_TEXT = new Set(["T", "TA", "TV", "TAV", "MISA"]);
const NEEDS_VIDEO = new Set(["A", "V", "TA", "TV", "AV", "TAV", "MISA"]);

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
    <main style={{ maxWidth: 640, margin: "48px auto", padding: "0 16px" }}>
      <h1>Multimodal Intent Inference</h1>
      <p className="muted">
        Every model here is honestly labeled with its own real, measured
        accuracy (Milestones 3 and 5) — text-only wins in every controlled
        comparison this project has run so far. This pipeline has no
        speech-to-text: type what was said yourself, even for models that
        also use audio/video.
      </p>

      <form onSubmit={handleSubmit} className="card" style={{ marginTop: 16 }}>
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

        {NEEDS_TEXT.has(modelChoice) && (
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
        )}

        {NEEDS_VIDEO.has(modelChoice) && (
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
        )}

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict intent"}
        </button>
      </form>

      {error && (
        <div className="error" style={{ marginTop: 20 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="card" style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p>
            <strong>Predicted intent:</strong> {result.predicted_intent}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
          </p>
          <p className="muted">{result.explanation.calibration_caveat}</p>
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
