const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const apiBaseInput = document.getElementById("apiBase");
const urlInput = document.getElementById("urlInput");
const resultElement = document.getElementById("result");

browser.storage.local.get({ apiBase: DEFAULT_API_BASE }).then(({ apiBase }) => {
  apiBaseInput.value = apiBase;
}).catch((error) => {
  resultElement.textContent = `Could not load API settings: ${error.message}`;
});

apiBaseInput.addEventListener("change", async () => {
  try {
    const apiBase = validateApiBase(apiBaseInput.value);
    apiBaseInput.value = apiBase;
    await browser.storage.local.set({ apiBase });
    resultElement.textContent = "API address saved.";
  } catch (error) {
    resultElement.textContent = error.message;
  }
});

document.getElementById("useTab").addEventListener("click", async () => {
  try {
    const [tab] = await browser.tabs.query({
      active: true,
      currentWindow: true,
    });
    urlInput.value = tab.url ?? "";
  } catch {
    resultElement.textContent = "Could not read the current tab URL.";
  }
});

document.getElementById("checkBtn").addEventListener("click", async () => {
  const url = urlInput.value.trim();
  if (!url) {
    resultElement.textContent = "Enter a URL to check.";
    return;
  }

  let apiBase;
  try {
    apiBase = validateApiBase(apiBaseInput.value);
  } catch (error) {
    resultElement.textContent = error.message;
    return;
  }

  const apiOrigin = new URL(apiBase).origin;
  const permission = browser.permissions.request({
    origins: [`${apiOrigin}/*`],
  });
  resultElement.textContent = "Checking...";

  try {
    if (!(await permission)) {
      resultElement.textContent = "Permission to contact the API was denied.";
      return;
    }
    await browser.storage.local.set({ apiBase });
    const response = await fetch(`${apiBase}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail ?? "The URL could not be checked.");
    }
    if (payload.label !== "good" && payload.label !== "bad") {
      throw new Error("The API returned an unexpected response.");
    }
    resultElement.textContent =
      payload.label === "good"
        ? "No obvious phishing signals found. Stay cautious."
        : "This link looks suspicious. Avoid opening it.";
  } catch (error) {
    resultElement.textContent = `Error: ${error.message}`;
  }
});

function validateApiBase(value) {
  const base = value.trim().replace(/\/+$/, "");
  let parsed;
  try {
    parsed = new URL(base);
  } catch {
    throw new Error("Enter the base URL of your prediction API.");
  }
  if (
    (parsed.protocol !== "https:" &&
      !(parsed.protocol === "http:" &&
        ["127.0.0.1", "localhost"].includes(parsed.hostname))) ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error("Use an HTTPS API origin (or localhost for development).");
  }
  return parsed.origin;
}
