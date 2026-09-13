// Entroly Chrome Extension - GitHub Content Script
// Injects the "⚡ Compress with Entroly" action into GitHub repository pages.

(function () {
  'use strict';

  function getRepoDetails() {
    const parts = window.location.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) {
      return {
        owner: parts[0],
        repo: parts[1],
        full: `${parts[0]}/${parts[1]}`
      };
    }
    return null;
  }

  function injectEntrolyButton() {
    const repoInfo = getRepoDetails();
    if (!repoInfo) return;

    // Avoid duplicate injection
    if (document.getElementById('entroly-github-btn')) return;

    // Look for GitHub repository actions bar or Clone button container
    // Common selectors across GitHub layout versions:
    const targetContainer = 
      document.querySelector('.pagehead-actions') || 
      document.querySelector('#repository-details-container ul') ||
      document.querySelector('[data-testid="latest-commit-details"]')?.parentElement ||
      document.querySelector('.file-navigation');

    if (!targetContainer) return;

    const li = document.createElement('li');
    li.id = 'entroly-github-btn-wrap';
    li.style.display = 'inline-block';
    li.style.marginRight = '8px';

    const btn = document.createElement('a');
    btn.id = 'entroly-github-btn';
    btn.className = 'btn btn-sm';
    btn.style.cssText = `
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
      color: #ffffff !important;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35);
      transition: all 0.2s ease;
    `;
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
      <span>⚡ Entroly</span>
    `;

    btn.title = `Compress ${repoInfo.full} with Entroly 0-1 Knapsack Context Engine`;

    btn.addEventListener('click', (e) => {
      e.preventDefault();
      // Notify extension background or open Web Playground pre-loaded with this repo
      chrome.runtime.sendMessage({
        type: 'OPEN_PLAYGROUND_WITH_REPO',
        repo: repoInfo.full
      });
    });

    if (targetContainer.tagName.toLowerCase() === 'ul') {
      targetContainer.prepend(li);
      li.appendChild(btn);
    } else {
      targetContainer.prepend(btn);
    }
  }

  // Run on initial load and GitHub SPA navigation (turbo:render / pjax:end)
  injectEntrolyButton();
  document.addEventListener('turbo:render', injectEntrolyButton);
  document.addEventListener('pjax:end', injectEntrolyButton);
  window.addEventListener('popstate', injectEntrolyButton);
})();
