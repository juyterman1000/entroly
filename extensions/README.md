# Entroly Browser Extensions

Official browser extensions for Google Chrome and Mozilla Firefox that add 1-click context compression and Merkle receipts directly to GitHub repository pages.

---

## Features

- **⚡ GitHub Header Action**: Adds an `⚡ Entroly` button on every GitHub repo page to immediately analyze and compress the repository.
- **Right-Click Context Menu**: Select any code on GitHub or web docs, right-click, and choose **"⚡ Compress Selection with Entroly"**.
- **Extension Popup**: Instant token count estimate, savings breakdown, and 1-click links to the hosted Web Playground.

---

## Local Development & Testing

### Google Chrome (or Brave / Edge / Arc)
1. Open `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top right corner).
3. Click **Load unpacked**.
4. Select the `extensions/chrome/` directory.
5. Navigate to any GitHub repository (e.g. `https://github.com/juyterman1000/entroly`). You will see the `⚡ Entroly` button injected next to the repo action bar.

### Mozilla Firefox
1. Open `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on...**
3. Select `extensions/firefox/manifest.json`.

---

## Web Store Distribution Checklist

- [x] Manifest V3 compliant
- [x] Content security policy verified (zero remote script execution)
- [x] Scoped permissions (`activeTab`, `contextMenus`, `storage`, `https://github.com/*`)
- [x] Privacy policy aligned ([PRIVACY.md](../../PRIVACY.md))
