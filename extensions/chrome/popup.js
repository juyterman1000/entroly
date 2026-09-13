// Entroly Chrome Extension - Popup Logic

const PLAYGROUND_URL = "https://juyterman1000.github.io/entroly/playground/";

document.addEventListener('DOMContentLoaded', async () => {
  let activeRepo = null;

  // Query active tab to see if it is a GitHub repo
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      const url = new URL(tab.url);
      if (url.hostname === 'github.com') {
        const parts = url.pathname.split('/').filter(Boolean);
        if (parts.length >= 2) {
          activeRepo = `${parts[0]}/${parts[1]}`;
          document.getElementById('repoName').textContent = activeRepo;
        }
      }
    }
  } catch (err) {
    console.error("Could not query tab:", err);
  }

  if (!activeRepo) {
    document.getElementById('repoName').textContent = "Not on a GitHub repo (general mode)";
  }

  // Open Playground
  document.getElementById('openPlaygroundBtn').addEventListener('click', () => {
    let target = PLAYGROUND_URL;
    if (activeRepo) {
      target += `?repo=${encodeURIComponent(activeRepo)}`;
    }
    chrome.tabs.create({ url: target });
  });

  // Copy Repo Link
  document.getElementById('copyRepoMarkdownBtn').addEventListener('click', () => {
    const text = activeRepo
      ? `[${activeRepo} on Entroly Playground](${PLAYGROUND_URL}?repo=${encodeURIComponent(activeRepo)})`
      : `[Entroly Context Compression](${PLAYGROUND_URL})`;

    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copyRepoMarkdownBtn');
      btn.textContent = "✓ Copied Markdown!";
      setTimeout(() => {
        btn.textContent = "📋 Copy Repo Context Link";
      }, 1500);
    });
  });
});
