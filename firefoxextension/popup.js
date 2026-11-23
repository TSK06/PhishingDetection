const API = "http://127.0.0.1:8000/predict";

document.getElementById('useTab').addEventListener('click', async () => {
  const [tab] = await browser.tabs.query({active: true, currentWindow: true});
  document.getElementById('urlInput').value = tab.url;
});

document.getElementById('checkBtn').addEventListener('click', async () => {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) return;

  document.getElementById('result').textContent = "Checking...";
  try {
    const r = await fetch(API, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url})
    });
    const j = await r.json();
    if (j.label === "good") {
      document.getElementById('result').textContent = "✅ Safe";
    } else {
      document.getElementById('result').textContent = "❌ Malicious";
    }
  } catch (e) {
    document.getElementById('result').textContent = "Error: " + e.message;
  }
});
