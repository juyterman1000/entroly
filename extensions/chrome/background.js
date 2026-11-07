// Entroly Chrome Extension - Service Worker (Background)

const PLAYGROUND_URL = "https://juyterman1000.github.io/entroly/playground/";

// Create Context Menu item on installation
chrome.runtime.onInstalled.addListener(() => {
  // Earlier versions left the last compressed selection here indefinitely.
  chrome.storage.local.remove("pendingCompressionText");
  chrome.contextMenus.create({
    id: "entroly-compress-selection",
    title: "⚡ Compress Selection with Entroly",
    contexts: ["selection"]
  });
});

// Handle Context Menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "entroly-compress-selection" && info.selectionText) {
    // The playground is a web page and cannot read extension storage, so the
    // selection used to be saved where nothing ever read it. It now travels
    // in the URL fragment, which the browser does not send to the server.
    chrome.tabs.create({
      url: `${PLAYGROUND_URL}#text=${encodeURIComponent(info.selectionText)}`
    });
  }
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'OPEN_PLAYGROUND_WITH_REPO') {
    const repo = encodeURIComponent(message.repo);
    chrome.tabs.create({
      url: `${PLAYGROUND_URL}?repo=${repo}`
    });
    sendResponse({ success: true });
  }
});
