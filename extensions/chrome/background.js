// Entroly Chrome Extension - Service Worker (Background)

const PLAYGROUND_URL = "https://juyterman1000.github.io/entroly/playground/";

// Create Context Menu item on installation
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "entroly-compress-selection",
    title: "⚡ Compress Selection with Entroly",
    contexts: ["selection"]
  });
});

// Handle Context Menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "entroly-compress-selection" && info.selectionText) {
    const textToCompress = info.selectionText;
    // Store selected text temporarily and open popup or playground
    chrome.storage.local.set({ pendingCompressionText: textToCompress }, () => {
      chrome.tabs.create({
        url: `${PLAYGROUND_URL}?mode=selection`
      });
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
