"use client";

import { useEffect, useState, type FormEvent } from "react";

type Prediction = {
  label: "good" | "bad";
};

type CheckRecord = {
  url: string;
  label: Prediction["label"];
  checkedAt: string;
};

function isCheckRecord(value: unknown): value is CheckRecord {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return (
    "url" in value &&
    typeof value.url === "string" &&
    "label" in value &&
    (value.label === "good" || value.label === "bad") &&
    "checkedAt" in value &&
    typeof value.checkedAt === "string"
  );
}

function getHostname(value: string): string {
  try {
    return new URL(value).hostname;
  } catch {
    return value;
  }
}

const STORAGE_KEY = "linklens-check-history";
const EXAMPLES = [
  "https://www.wikipedia.org",
  "http://secure-login-google.com",
];
const REQUEST_TIMEOUT_MS = 20_000;

export default function Home() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<CheckRecord | null>(null);
  const [history, setHistory] = useState<CheckRecord[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: unknown = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          setHistory(parsed.filter(isCheckRecord).slice(0, 20));
        }
      }
    } catch (caught) {
      console.warn("Unable to read LinkLens check history.", caught);
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Storage may be unavailable in restricted browser contexts.
      }
    }
  }, []);

  async function checkUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) return;

    setError("");
    setResult(null);
    setLoading(true);

    const controller = new AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(),
      REQUEST_TIMEOUT_MS,
    );

    try {
      const configuredApiBase =
        process.env.NEXT_PUBLIC_PREDICTION_API_URL?.replace(/\/+$/, "") ?? "";
      // `api/predict.py` is served by Vercel, not by `next dev`.
      const apiBase =
        process.env.NODE_ENV === "development" &&
        (!configuredApiBase || configuredApiBase === window.location.origin)
          ? "http://127.0.0.1:8000"
          : configuredApiBase;
      const response = await fetch(`${apiBase}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
        signal: controller.signal,
      });

      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        throw new Error(
          "The prediction service returned an unreadable response. Please try again.",
        );
      }

      if (!response.ok) {
        const message =
          typeof payload === "object" &&
          payload !== null &&
          "detail" in payload &&
          typeof payload.detail === "string"
            ? payload.detail
            : "The URL could not be checked. Please review it and try again.";
        throw new Error(message);
      }
      if (
        typeof payload !== "object" ||
        payload === null ||
        !("label" in payload) ||
        (payload.label !== "good" && payload.label !== "bad")
      ) {
        throw new Error(
          "The prediction service returned an unexpected result. Please try again.",
        );
      }

      const record: CheckRecord = {
        url: url.trim(),
        label: payload.label,
        checkedAt: new Date().toISOString(),
      };
      const nextHistory = [record, ...history].slice(0, 20);
      setResult(record);
      setHistory(nextHistory);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(nextHistory));
      } catch (caught) {
        console.warn("Unable to save LinkLens check history.", caught);
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        setError("The check took too long. Check your connection and try again.");
      } else if (caught instanceof TypeError) {
        setError(
          "Could not connect to the prediction service. Check your connection and try again.",
        );
      } else {
        setError(
          caught instanceof Error
            ? caught.message
            : "The URL could not be checked. Please review it and try again.",
        );
      }
    } finally {
      window.clearTimeout(timeoutId);
      setLoading(false);
    }
  }

  function updateUrl(value: string) {
    setUrl(value);
    setError("");
    setResult(null);
  }

  function clearHistory() {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (caught) {
      console.warn("Unable to clear LinkLens check history.", caught);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="LinkLens home">
          <span className="brand-mark" aria-hidden="true">
            L
          </span>
          <span>linklens</span>
        </a>
        <span className="topbar-note">URL phishing check</span>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">A second look before you click</p>
        <h1 id="page-title">
          Not every link
          <br />
          <span>is what it seems.</span>
        </h1>
        <p className="hero-copy">
          Check a URL for phishing signals with a machine-learning model. Use
          the result as one signal, not a guarantee of safety.
        </p>
      </section>

      <section
        className="checker-card"
        aria-labelledby="checker-title"
        aria-busy={loading}
      >
        <div className="card-heading">
          <div>
            <p className="eyebrow">URL checker</p>
            <h2 id="checker-title">Scan a link</h2>
            <p className="card-description">
              Review the address before opening it.
            </p>
          </div>
          <span className="protocol-badge">HTTP · HTTPS</span>
        </div>

        <form onSubmit={checkUrl}>
          <label className="field-label" htmlFor="url">
            URL to inspect
          </label>
          <div className="input-row">
            <span className="input-icon" aria-hidden="true">
              ↗
            </span>
            <input
              id="url"
              type="url"
              inputMode="url"
              autoComplete="url"
              placeholder="https://example.com/page"
              value={url}
              onChange={(event) => updateUrl(event.target.value)}
              maxLength={2048}
              aria-describedby="url-hint"
              disabled={loading}
              required
            />
            <button className="check-button" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="button-spinner" aria-hidden="true" />
                  Checking
                </>
              ) : (
                <>
                  Check link <span aria-hidden="true">→</span>
                </>
              )}
            </button>
          </div>
          <p className="field-hint" id="url-hint">
            Include the full address, starting with http:// or https://.
          </p>
        </form>

        <div className="examples" aria-label="Example URLs">
          <span className="examples-label">Try an example</span>
          {EXAMPLES.map((example) => (
            <button
              className="example-chip"
              key={example}
              type="button"
              disabled={loading}
              onClick={() => updateUrl(example)}
            >
              {getHostname(example)}
            </button>
          ))}
        </div>

        <div className="feedback-area" aria-busy={loading}>
          {loading ? (
            <div className="feedback-panel feedback-loading" role="status">
              <span className="feedback-symbol loading-symbol" aria-hidden="true">
                <span className="loading-spinner" />
              </span>
              <div>
                <strong>Checking this URL</strong>
                <p>The prediction service is reviewing the address.</p>
              </div>
            </div>
          ) : error ? (
            <div className="feedback-panel feedback-error" role="alert">
              <span className="feedback-symbol" aria-hidden="true">
                !
              </span>
              <div>
                <strong>We couldn’t complete the check</strong>
                <p>{error}</p>
                <p className="recovery-copy">
                  Review the address above, then try again.
                </p>
              </div>
            </div>
          ) : result ? (
            <div
              className={`feedback-panel result-panel ${result.label === "good" ? "result-good" : "result-bad"}`}
              role="status"
              aria-live="polite"
              aria-atomic="true"
            >
              <span className="feedback-symbol" aria-hidden="true">
                {result.label === "good" ? "✓" : "!"}
              </span>
              <div className="result-content">
                <p className="result-kicker">
                  {result.label === "good" ? "No obvious signals found" : "Potential phishing signals"}
                </p>
                <strong>
                  {result.label === "good"
                    ? "This URL was classified as likely safe"
                    : "This URL was classified as suspicious"}
                </strong>
                <p className="result-guidance">
                  {result.label === "good"
                    ? "A model result cannot guarantee that a site is safe. Check the address before sharing information."
                    : "Avoid opening it or entering passwords and payment details."
                  }
                </p>
                <div className="result-address">
                  <span>Scanned address</span>
                  <code title={result.url}>{result.url}</code>
                </div>
              </div>
            </div>
          ) : (
            <div className="feedback-placeholder" aria-live="polite">
              <span className="placeholder-mark" aria-hidden="true">
                ↳
              </span>
              <span>Your scan result will appear here.</span>
            </div>
          )}
        </div>
      </section>

      <section className="trust-row" aria-label="About this check">
        <div className="trust-item">
          <span className="trust-icon" aria-hidden="true">
            01
          </span>
          <p>
            <strong>Model-based signal</strong>
            <span>Checks URL text, not the live website.</span>
          </p>
        </div>
        <div className="trust-item">
          <span className="trust-icon" aria-hidden="true">
            02
          </span>
          <p>
            <strong>Stay cautious</strong>
            <span>A likely-safe result is not a guarantee.</span>
          </p>
        </div>
        <div className="trust-item">
          <span className="trust-icon" aria-hidden="true">
            03
          </span>
          <p>
            <strong>Recent checks</strong>
            <span>Stored locally in this browser.</span>
          </p>
        </div>
      </section>

      <section className="history-section" aria-labelledby="history-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">On this device</p>
            <h2 id="history-title">Recent checks</h2>
          </div>
          {history.length > 0 && (
            <button
              className="text-button"
              type="button"
              onClick={clearHistory}
            >
              Clear history
            </button>
          )}
        </div>
        {history.length === 0 ? (
          <p className="empty-state">
            Your recent URL checks will be listed here.
          </p>
        ) : (
          <ul className="history-list">
            {history.slice(0, 5).map((record) => (
              <li key={`${record.checkedAt}-${record.url}`}>
                <span
                  className={`history-indicator ${record.label === "good" ? "indicator-good" : "indicator-bad"}`}
                  aria-hidden="true"
                />
                <span className="history-label">
                  {record.label === "good" ? "Likely safe" : "Suspicious"}
                </span>
                <span className="history-url" title={record.url}>
                  {record.url}
                </span>
                <time className="history-time" dateTime={record.checkedAt}>
                  {new Date(record.checkedAt).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
              </li>
            ))}
          </ul>
        )}
      </section>

      <footer className="footer">
        <span>LinkLens · Phishing URL detection</span>
        <span>When in doubt, don’t click.</span>
      </footer>
    </main>
  );
}
