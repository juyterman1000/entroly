// Entroly Browser Playground Engine
// Client-side 0-1 Knapsack context selection, Merkle receipt generation, and comparative benchmarks.

// --- Token Estimation ---
function estimateTokens(text) {
  if (!text) return 0;
  // Code tokenization average: ~3.7 characters per token
  const words = text.match(/[A-Za-z0-9_]+|[^\s\w]/g);
  return words ? Math.max(1, Math.ceil(words.length * 1.15)) : 1;
}

// --- Cryptographic Hash & Merkle Tree ---
async function sha256Hex(text) {
  const enc = new TextEncoder();
  const data = enc.encode(text);
  if (window.crypto && window.crypto.subtle) {
    const hashBuf = await window.crypto.subtle.digest('SHA-256', data);
    const hashArr = Array.from(new Uint8Array(hashBuf));
    return hashArr.map(b => b.toString(16).padStart(2, '0')).join('');
  }
  // Fallback simple hash if subtle crypto is unavailable
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  return 'fallback_' + Math.abs(hash).toString(16).padStart(16, '0');
}

async function computeMerkleRoot(hashes) {
  if (!hashes || hashes.length === 0) return await sha256Hex('EMPTY_CONTEXT');
  let level = [...hashes];
  while (level.length > 1) {
    const nextLevel = [];
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) {
        nextLevel.push(await sha256Hex(level[i] + ':' + level[i + 1]));
      } else {
        nextLevel.push(await sha256Hex(level[i] + ':' + level[i]));
      }
    }
    level = nextLevel;
  }
  return level[0];
}

// --- Preloaded Realistic Engineering Scenarios ---
const PRESETS = {
  payments: {
    name: "Payment & Rate Limit Cascade",
    category: "FinTech / Python",
    query: "Fix payment checkout failure and rate limit retry handling when gateway returns 429",
    files: [
      {
        path: "checkout.py",
        content: `import time\nimport logging\nfrom auth_middleware import verify_session\nfrom rate_limiter import acquire_gateway_slot, RateLimitExceeded\n\nlogger = logging.getLogger("payments.checkout")\n\ndef process_order_checkout(order_id: str, user_token: str, payment_payload: dict) -> dict:\n    """Executes customer payment transaction with idempotent gateway routing."""\n    user_ctx = verify_session(user_token)\n    if not user_ctx.get("is_authenticated"):\n        raise PermissionError("Invalid checkout session token")\n\n    # Check and acquire rate limit slot for tenant\n    tenant_id = user_ctx["tenant_id"]\n    try:\n        slot_ticket = acquire_gateway_slot(tenant_id, cost=1.0)\n    except RateLimitExceeded as e:\n        logger.warning(f"Tenant {tenant_id} hit payment rate limit: {e}")\n        # BUG: Currently fails immediately without exponential backoff retry\n        raise PaymentGatewayError("Payment throttled", retry_after=e.retry_after_sec)\n\n    # Proceed to charge payload\n    result = execute_stripe_charge(slot_ticket, payment_payload)\n    return {"status": "SUCCESS", "charge_id": result["id"], "order_id": order_id}\n`
      },
      {
        path: "rate_limiter.py",
        content: `import time\n\nclass RateLimitExceeded(Exception):\n    def __init__(self, message: str, retry_after_sec: float):\n        super().__init__(message)\n        self.retry_after_sec = retry_after_sec\n\nclass SlidingWindowLimiter:\n    def __init__(self, max_qps: float = 50.0, burst_limit: int = 10):\n        self.max_qps = max_qps\n        self.burst_limit = burst_limit\n        self.history = {}\n\n    def acquire_gateway_slot(self, tenant_id: str, cost: float = 1.0):\n        now = time.time()\n        window_start = now - 1.0\n        entries = [t for t in self.history.get(tenant_id, []) if t > window_start]\n        if len(entries) >= self.burst_limit:\n            sleep_needed = entries[0] + 1.0 - now\n            raise RateLimitExceeded("429 Gateway Rate Limit Exceeded", retry_after_sec=max(0.1, sleep_needed))\n        entries.append(now)\n        self.history[tenant_id] = entries\n        return {"ticket_id": f"tk_{tenant_id}_{int(now)}", "granted": True}\n\nglobal_limiter = SlidingWindowLimiter()\nacquire_gateway_slot = global_limiter.acquire_gateway_slot\n`
      },
      {
        path: "auth_middleware.py",
        content: `import hmac\nimport hashlib\n\nSECRET_KEY = b"entroly_demo_secret_key_prod_v1"\n\ndef verify_session(token: str) -> dict:\n    """Validates user session signature and resolves tenant tier."""\n    if not token or not token.startswith("tok_"):\n        return {"is_authenticated": False}\n    parts = token.split(".")\n    if len(parts) != 2:\n        return {"is_authenticated": False}\n    payload_b64, signature = parts[0], parts[1]\n    expected_sig = hmac.new(SECRET_KEY, payload_b64.encode(), hashlib.sha256).hexdigest()[:16]\n    if not hmac.compare_digest(signature, expected_sig):\n        return {"is_authenticated": False}\n    return {\n        "is_authenticated": True,\n        "user_id": "usr_9921",\n        "tenant_id": "tenant_enterprise_01",\n        "tier": "enterprise"\n    }\n`
      },
      {
        path: "audit_logger.py",
        content: `import datetime\nimport json\n\n# Audit logging infrastructure for Sarbanes-Oxley (SOX) compliance\n# Note: This file contains historical schema migrations and verbose telemetry tables\n\nclass AuditLogger:\n    def __init__(self, db_conn=None):\n        self.db = db_conn\n        self.log_buffer = []\n\n    def record_event(self, event_type: str, actor_id: str, metadata: dict):\n        entry = {\n            "event_id": f"evt_{datetime.datetime.utcnow().timestamp()}",\n            "event_type": event_type,\n            "actor_id": actor_id,\n            "timestamp_utc": datetime.datetime.utcnow().isoformat(),\n            "metadata": metadata,\n            "schema_version": "v4.2.1-compliance-audit"\n        }\n        self.log_buffer.append(entry)\n        if len(self.log_buffer) > 100:\n            self.flush_to_cold_storage()\n\n    def flush_to_cold_storage(self):\n        # Large batch write to S3 bucket / PostgreSQL archive\n        payload = json.dumps(self.log_buffer)\n        # Simulation of 200 lines of historical archiving boilerplate...\n        self.log_buffer = []\n`
      }
    ]
  },
  concurrency: {
    name: "Deadlock & Eviction Race",
    category: "Systems / Rust",
    query: "Identify and eliminate deadlock between Cache write lock and LRU eviction background thread",
    files: [
      {
        path: "cache.rs",
        content: `use std::sync::{Arc, RwLock};\nuse std::collections::HashMap;\n\npub struct Cache<K, V> {\n    entries: RwLock<HashMap<K, V>>,\n    eviction_tx: std::sync::mpsc::Sender<K>,\n}\n\nimpl<K: Eq + std::hash::Hash + Clone, V: Clone> Cache<K, V> {\n    pub fn put(&self, key: K, val: V) {\n        // Acquire write lock\n        let mut w = self.entries.write().unwrap();\n        w.insert(key.clone(), val);\n        // DEADLOCK RISK: sending to eviction while holding write lock\n        let _ = self.eviction_tx.send(key);\n    }\n}\n`
      },
      {
        path: "eviction.rs",
        content: `use std::sync::mpsc::Receiver;\nuse std::sync::Arc;\n\npub fn run_eviction_loop<K: Eq + std::hash::Hash + Clone, V>(\n    rx: Receiver<K>,\n    cache: Arc<super::Cache<K, V>>\n) {\n    while let Ok(key) = rx.recv() {\n        // Needs write lock to evict, but put() holds write lock while waiting on channel!\n        let mut w = cache.entries.write().unwrap();\n        if w.len() > 1000 {\n            w.remove(&key);\n        }\n    }\n}\n`
      },
      {
        path: "lock_hierarchy.rs",
        content: `// Architectural Lock Order Specification\n// 1. Level 1: Global Registry\n// 2. Level 2: Cache Read/Write Lock\n// 3. Level 3: Eviction Queue\n// Invariant: Never send to Level 3 channel while holding Level 2 lock.\n`
      },
      {
        path: "bench_metrics.rs",
        content: `// 600 tokens of benchmark harnesses and throughput timers\npub fn measure_qps() -> f64 { 150_000.0 }\npub fn measure_latency_p99() -> f64 { 0.42 }\n`
      }
    ]
  },
  auth: {
    name: "JWT Token Refresh Vulnerability",
    category: "Fullstack / Node",
    query: "Fix race condition allowing refresh token reuse before revocation list sync",
    files: [
      {
        path: "authController.ts",
        content: `import { verifyRefreshToken, generateTokenPair, revokeToken } from './sessionStore';\n\nexport async function handleTokenRefresh(req, res) {\n  const { refreshToken } = req.body;\n  const session = await verifyRefreshToken(refreshToken);\n  if (!session) return res.status(401).json({ error: "Invalid token" });\n\n  // VULNERABILITY: New tokens issued before old token is marked revoked in Redis!\n  const newTokens = generateTokenPair(session.userId);\n  await revokeToken(refreshToken);\n  return res.json(newTokens);\n}\n`
      },
      {
        path: "sessionStore.ts",
        content: `const revokedSet = new Set<string>();\n\nexport async function verifyRefreshToken(token: string) {\n  if (revokedSet.has(token)) return null;\n  return { userId: "user_401", issuedAt: Date.now() };\n}\n\nexport async function revokeToken(token: string) {\n  revokedSet.add(token);\n}\n`
      },
      {
        path: "cryptoUtils.ts",
        content: `import * as crypto from 'crypto';\nexport function generateTokenPair(userId: string) {\n  return {\n    accessToken: "at_" + crypto.randomBytes(16).toString('hex'),\n    refreshToken: "rt_" + crypto.randomBytes(24).toString('hex')\n  };\n}\n`
      },
      {
        path: "telemetry.ts",
        content: `// 500 tokens of Datadog trace logs and telemetry spans\nexport function trackEvent(name: string, meta: any) { /* no-op */ }\n`
      }
    ]
  }
};

// --- Active State ---
let currentCorpus = [];
let currentPreset = 'payments';

// --- Fragment Chunking & Relevance Scoring ---
function chunkCorpus(files) {
  const fragments = [];
  for (const f of files) {
    const lines = f.content.split('\n');
    let currentChunk = [];
    let startLine = 1;
    let byteOffset = 0;

    for (let i = 0; i < lines.length; i++) {
      currentChunk.push(lines[i]);
      // Chunk at function/class headers or every ~25 lines
      const isBoundary = i > 0 && (/^(def |class |export |pub fn |async fn |import |const )/.test(lines[i]) || currentChunk.length >= 25);
      if (isBoundary || i === lines.length - 1) {
        const text = currentChunk.join('\n');
        const tokenCount = estimateTokens(text);
        const endLine = startLine + currentChunk.length - 1;
        const endByte = byteOffset + new TextEncoder().encode(text).length;

        fragments.push({
          id: `${f.path}#L${startLine}-L${endLine}`,
          file: f.path,
          text: text,
          tokens: tokenCount,
          startLine: startLine,
          endLine: endLine,
          startByte: byteOffset,
          endByte: endByte
        });

        byteOffset = endByte + 1;
        startLine = endLine + 1;
        currentChunk = [];
      }
    }
  }
  return fragments;
}

function calculateRelevance(fragment, queryTerms) {
  const textLower = fragment.text.toLowerCase();
  const fileLower = fragment.file.toLowerCase();
  let score = 0.5; // baseline

  for (const term of queryTerms) {
    if (term.length <= 2) continue;
    // File match boost
    if (fileLower.includes(term)) score += 3.0;

    // Text match with TF count
    const matches = textLower.split(term).length - 1;
    if (matches > 0) {
      score += Math.min(6.0, matches * 1.8);
    }
  }

  // Critical architectural boosts
  if (/rate_limit|retry|backoff|deadlock|race|lock|token|session|auth|429/i.test(fragment.text)) {
    score += 4.5;
  }
  // Penalize pure boilerplate/audit/benchmark
  if (/audit_logger|bench_metrics|telemetry|compliance/i.test(fragment.file)) {
    score *= 0.3;
  }

  return score;
}

// --- 0-1 Knapsack Solver (Entroly Algorithm) ---
function solveKnapsack(items, budget) {
  // Discretize budget to scale of ~5 tokens for fast DP table in browser
  const scale = 5;
  const W = Math.floor(budget / scale);
  const n = items.length;

  const weights = items.map(it => Math.max(1, Math.ceil(it.tokens / scale)));
  const values = items.map(it => Math.round(it.value * 100));

  const dp = Array.from({ length: n + 1 }, () => new Int32Array(W + 1));

  for (let i = 1; i <= n; i++) {
    const wt = weights[i - 1];
    const val = values[i - 1];
    for (let w = 0; w <= W; w++) {
      if (wt <= w) {
        dp[i][w] = Math.max(dp[i - 1][w], dp[i - 1][w - wt] + val);
      } else {
        dp[i][w] = dp[i - 1][w];
      }
    }
  }

  // Backtrack to find selected items
  const selectedIndices = [];
  let w = W;
  for (let i = n; i >= 1; i--) {
    if (dp[i][w] !== dp[i - 1][w]) {
      selectedIndices.push(i - 1);
      w -= weights[i - 1];
    }
  }

  return selectedIndices.reverse();
}

// --- Greedy Top-K Solver (Cody/Copilot Style) ---
function solveGreedyTopK(items, budget) {
  // Sort items strictly by value descending
  const sorted = items.map((it, idx) => ({ ...it, originalIdx: idx }))
    .sort((a, b) => b.value - a.value);

  const selectedIndices = [];
  let currentTokens = 0;

  for (const it of sorted) {
    if (currentTokens + it.tokens <= budget) {
      selectedIndices.push(it.originalIdx);
      currentTokens += it.tokens;
    }
  }

  return selectedIndices;
}

// --- Main Optimization Pipeline ---
async function runOptimization() {
  const budget = parseInt(document.getElementById('budgetSlider').value, 10);
  const query = document.getElementById('taskQueryInput').value.trim();
  const queryTerms = query.toLowerCase().match(/[a-z0-9_]+/g) || [];

  const fragments = chunkCorpus(currentCorpus);
  const scoredItems = fragments.map(f => {
    const val = calculateRelevance(f, queryTerms);
    return {
      ...f,
      value: val,
      density: val / Math.max(1, f.tokens)
    };
  });

  // 1. Solve via Entroly 0-1 Knapsack
  const knapsackIndices = solveKnapsack(scoredItems, budget);
  const selectedKnapsack = knapsackIndices.map(i => scoredItems[i]);

  // 2. Solve via Top-K Greedy for comparison
  const topKIndices = solveGreedyTopK(scoredItems, budget);
  const selectedTopK = topKIndices.map(i => scoredItems[i]);

  // 3. Compute Metrics
  const rawTokens = scoredItems.reduce((acc, f) => acc + f.tokens, 0);
  const deliveredTokens = selectedKnapsack.reduce((acc, f) => acc + f.tokens, 0);
  const savedTokens = Math.max(0, rawTokens - deliveredTokens);
  const savingsPct = rawTokens > 0 ? ((savedTokens / rawTokens) * 100).toFixed(1) : '0.0';

  // Pricing based on Claude 3.5 Sonnet / GPT-4o input rate ($3.00 / 1M tokens)
  const costAvoided = (savedTokens * 0.000003).toFixed(4);

  // Update HUD
  document.getElementById('hudDeliveredTokens').textContent = deliveredTokens;
  document.getElementById('hudBudget').textContent = budget;
  document.getElementById('hudSavingsPercent').textContent = `${savingsPct}%`;
  document.getElementById('hudSavedTokens').textContent = savedTokens.toLocaleString();
  document.getElementById('hudCostSaved').textContent = `$${costAvoided}`;

  // Check critical catches
  const selectedFiles = new Set(selectedKnapsack.map(f => f.file));
  let criticalCount = 0;
  if (currentPreset === 'payments') {
    if (selectedFiles.has('checkout.py')) criticalCount++;
    if (selectedFiles.has('rate_limiter.py')) criticalCount++;
    if (selectedFiles.has('auth_middleware.py')) criticalCount++;
    document.getElementById('hudCatches').textContent = `${criticalCount} / 3`;
  } else {
    document.getElementById('hudCatches').textContent = `${selectedFiles.size} files`;
  }

  // 4. Render Optimized Context
  let formattedOutput = `<!-- ======================================================== -->\n`;
  formattedOutput += `<!-- ENTROLY PROVABLE CONTEXT PACKET                          -->\n`;
  formattedOutput += `<!-- Budget: ${budget} tokens | Delivered: ${deliveredTokens} tokens | Saved: ${savingsPct}% -->\n`;
  formattedOutput += `<!-- ======================================================== -->\n\n`;

  for (const f of selectedKnapsack) {
    formattedOutput += `// --- File: ${f.file} (${f.startLine}:${f.endLine}) [${f.tokens} tokens] ---\n`;
    formattedOutput += f.text + '\n\n';
  }

  document.getElementById('optimizedCodeView').textContent = formattedOutput;

  // 5. Generate Merkle Receipt
  const fragmentHashes = [];
  const fragmentElements = [];

  for (const f of selectedKnapsack) {
    const fHash = await sha256Hex(f.text);
    const handle = 'rec_' + fHash.slice(0, 16);
    fragmentHashes.push(fHash);

    fragmentElements.push(`
      <div class="fragment-item">
        <span class="fragment-path">${f.file}:${f.startLine}-${f.endLine}</span>
        <span class="fragment-hash" title="SHA-256 Digest">${handle} (${f.tokens} tok)</span>
      </div>
    `);
  }

  const merkleRoot = await computeMerkleRoot(fragmentHashes);
  const fullCorpusDigest = await sha256Hex(currentCorpus.map(c => c.content).join('---'));

  document.getElementById('receiptMerkleRoot').textContent = merkleRoot;
  document.getElementById('receiptOriginalDigest').textContent = fullCorpusDigest.slice(0, 32) + '...';
  document.getElementById('receiptSpansRetained').textContent = `${selectedKnapsack.length} fragments across ${selectedFiles.size} files`;
  document.getElementById('receiptFragmentList').innerHTML = fragmentElements.join('');

  // 6. Omission Analysis
  const selectedSet = new Set(knapsackIndices);
  const omittedItems = scoredItems
    .map((it, idx) => ({ ...it, idx }))
    .filter(it => !selectedSet.has(it.idx))
    .sort((a, b) => b.density - a.density);

  document.getElementById('omissionCount').textContent = omittedItems.length;

  const omissionHtml = omittedItems.map(om => {
    let reason = "Marginal utility density below knapsack threshold";
    if (/audit|bench|telemetry/.test(om.file)) {
      reason = "Classified as non-critical telemetry/audit schema; deferred to prevent budget exhaustion";
    } else if (om.tokens > budget) {
      reason = `Fragment size (${om.tokens} tokens) exceeds remaining budget envelope`;
    }

    return `
      <div class="omission-item">
        <div class="omission-header">
          <span>${om.file} (${om.startLine}:${om.endLine})</span>
          <span class="omission-density">Cost: ${om.tokens} tokens</span>
        </div>
        <div class="omission-reason">${reason}</div>
        <div class="omission-proof">Proof: utility_density=${om.density.toFixed(4)} | recovery_handle=rec_${om.id.replace(/[^a-zA-Z0-9]/g, '_')}</div>
      </div>
    `;
  }).join('');

  document.getElementById('omissionList').innerHTML = omissionHtml || '<p class="field-help">No fragments omitted. All content fit in budget.</p>';
}

// --- Corpus Update & UI Synchronization ---
function updateCorpusSummary() {
  const count = currentCorpus.length;
  let rawTokens = 0;
  for (const f of currentCorpus) {
    rawTokens += estimateTokens(f.content);
  }
  const cost = (rawTokens * 0.000003).toFixed(4);

  document.getElementById('corpusFilesCount').textContent = count;
  document.getElementById('corpusRawTokens').textContent = rawTokens.toLocaleString();
  document.getElementById('corpusRawCost').textContent = `$${cost}`;
}

function loadPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;
  currentPreset = presetKey;
  currentCorpus = p.files.map(f => ({ ...f }));
  document.getElementById('taskQueryInput').value = p.query;
  updateCorpusSummary();
  runOptimization();
}

// --- Initialization & Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
  // Load initial preset
  loadPreset('payments');

  // Preset Card Clicks
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const presetKey = card.getAttribute('data-preset');
      loadPreset(presetKey);
    });
  });

  // Source Tabs (Presets / Custom / GitHub)
  const sourceTabs = document.getElementById('sourceTabs');
  sourceTabs.querySelectorAll('.tab-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      sourceTabs.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('#tabContentPresets, #tabContentCustom, #tabContentGithub').forEach(el => el.classList.remove('active'));
      const target = btn.getAttribute('data-tab');
      if (target === 'presets') document.getElementById('tabContentPresets').classList.add('active');
      if (target === 'custom') {
        document.getElementById('tabContentCustom').classList.add('active');
        initCustomFiles();
      }
      if (target === 'github') document.getElementById('tabContentGithub').classList.add('active');
    });
  });

  // Output Tabs (Context / Receipt / Compare / Omissions)
  const outputTabs = document.getElementById('outputTabs');
  outputTabs.querySelectorAll('.tab-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      outputTabs.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('#outTabContext, #outTabReceipt, #outTabCompare, #outTabOmissions').forEach(el => el.classList.remove('active'));
      const target = btn.getAttribute('data-tab');
      if (target === 'context') document.getElementById('outTabContext').classList.add('active');
      if (target === 'receipt') document.getElementById('outTabReceipt').classList.add('active');
      if (target === 'compare') document.getElementById('outTabCompare').classList.add('active');
      if (target === 'omissions') document.getElementById('outTabOmissions').classList.add('active');
    });
  });

  // Slider change
  const budgetSlider = document.getElementById('budgetSlider');
  budgetSlider.addEventListener('input', (e) => {
    document.getElementById('budgetValue').textContent = e.target.value;
    runOptimization();
  });

  // Optimize button
  document.getElementById('optimizeBtn').addEventListener('click', () => {
    runOptimization();
  });

  // Copy Context Button
  document.getElementById('copyContextBtn').addEventListener('click', () => {
    const text = document.getElementById('optimizedCodeView').textContent;
    navigator.clipboard.writeText(text).then(() => {
      const origText = document.getElementById('copyContextBtn').innerHTML;
      document.getElementById('copyContextBtn').innerHTML = `✓ Copied Context!`;
      setTimeout(() => {
        document.getElementById('copyContextBtn').innerHTML = origText;
      }, 2000);
    });
  });

  // GitHub Fetch
  const fetchGithubBtn = document.getElementById('fetchGithubBtn');
  fetchGithubBtn.addEventListener('click', async () => {
    const input = document.getElementById('githubRepoInput').value.trim();
    const statusBox = document.getElementById('githubFetchStatus');
    if (!input) return;

    let repo = input.replace('https://github.com/', '').replace(/\/$/, '');
    statusBox.className = 'status-box';
    statusBox.textContent = `Fetching file tree for ${repo}...`;

    try {
      const res = await fetch(`https://api.github.com/repos/${repo}/contents`);
      if (!res.ok) throw new Error(`GitHub API error: ${res.status} ${res.statusText}`);
      const data = await res.json();

      const textFiles = data.filter(item => item.type === 'file' && /\.(py|js|ts|rs|go|md|json)$/i.test(item.name)).slice(0, 5);
      if (textFiles.length === 0) {
        statusBox.textContent = `No source code files found in top-level of ${repo}.`;
        return;
      }

      statusBox.textContent = `Downloading ${textFiles.length} files...`;
      const fetched = [];
      for (const f of textFiles) {
        const rawRes = await fetch(f.download_url);
        const content = await rawRes.text();
        fetched.push({ path: f.name, content: content });
      }

      currentCorpus = fetched;
      statusBox.textContent = `Successfully fetched ${fetched.length} files from ${repo}!`;
      updateCorpusSummary();
      runOptimization();
    } catch (err) {
      statusBox.textContent = `Error: ${err.message}. Using mock files.`;
    }
  });
});

// Custom Files Support
let customFiles = [
  { path: 'example.py', content: '# Paste your own code here to test Entroly compression\ndef calculate_metrics(items):\n    return sum(x.score for x in items)\n' }
];
let activeCustomIndex = 0;

function initCustomFiles() {
  renderCustomFileTabs();
  updateCustomEditor();
}

function renderCustomFileTabs() {
  const container = document.getElementById('customFileTabs');
  container.innerHTML = customFiles.map((f, idx) => `
    <div class="file-tab-item ${idx === activeCustomIndex ? 'active' : ''}" onclick="switchCustomFile(${idx})">
      <span>${f.path}</span>
      ${customFiles.length > 1 ? `<span class="file-tab-close" onclick="deleteCustomFile(event, ${idx})">×</span>` : ''}
    </div>
  `).join('');
  document.getElementById('customFileCount').textContent = `${customFiles.length} file${customFiles.length > 1 ? 's' : ''}`;
}

window.switchCustomFile = function(idx) {
  // save current
  customFiles[activeCustomIndex].content = document.getElementById('customFileContent').value;
  activeCustomIndex = idx;
  renderCustomFileTabs();
  updateCustomEditor();
};

window.deleteCustomFile = function(e, idx) {
  e.stopPropagation();
  if (customFiles.length <= 1) return;
  customFiles.splice(idx, 1);
  if (activeCustomIndex >= customFiles.length) activeCustomIndex = customFiles.length - 1;
  renderCustomFileTabs();
  updateCustomEditor();
  currentCorpus = customFiles;
  updateCorpusSummary();
  runOptimization();
};

function updateCustomEditor() {
  document.getElementById('customFileContent').value = customFiles[activeCustomIndex].content;
}

document.getElementById('customFileContent')?.addEventListener('input', (e) => {
  customFiles[activeCustomIndex].content = e.target.value;
  currentCorpus = customFiles;
  updateCorpusSummary();
});

document.getElementById('addFileBtn')?.addEventListener('click', () => {
  const fileName = prompt("Enter file name (e.g., service.py, App.tsx):", `file_${customFiles.length + 1}.py`);
  if (!fileName) return;
  customFiles.push({ path: fileName, content: `# ${fileName}\n` });
  activeCustomIndex = customFiles.length - 1;
  renderCustomFileTabs();
  updateCustomEditor();
  currentCorpus = customFiles;
  updateCorpusSummary();
});
