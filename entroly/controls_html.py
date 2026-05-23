"""Entroly Control Panel — Full-featured daemon control UI."""

CONTROLS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly — Control Panel</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#050508;--bg2:#0a0b10;--card:rgba(14,17,24,0.85);--glass:rgba(255,255,255,0.03);
--border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.12);--text:#e8ecf4;--dim:#6b7280;
--emerald:#34d399;--blue:#60a5fa;--violet:#a78bfa;--amber:#fbbf24;--rose:#fb7185;--cyan:#22d3ee;
--grad1:linear-gradient(135deg,#667eea,#764ba2);--grad2:linear-gradient(135deg,#34d399,#06b6d4);}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
body::before{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;
background:radial-gradient(circle at 30% 20%,rgba(102,126,234,0.04),transparent 50%),
radial-gradient(circle at 70% 80%,rgba(118,75,162,0.03),transparent 50%);z-index:0;pointer-events:none;}
.topbar{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
padding:14px 32px;background:rgba(5,5,8,0.85);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);}
.brand{display:flex;align-items:center;gap:14px;}
.brand h1{font-size:22px;font-weight:900;background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-links{display:flex;gap:6px;}
.nav-links a{padding:6px 14px;border-radius:8px;font-size:13px;font-weight:500;color:var(--dim);
text-decoration:none;transition:all .2s;}
.nav-links a:hover{background:var(--glass);color:var(--text);}
.nav-links a.active{background:rgba(102,126,234,0.15);color:#667eea;}
.status-pill{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:20px;
background:rgba(52,211,153,0.08);font-size:12px;font-weight:600;color:var(--emerald);}
.status-pill .dot{width:7px;height:7px;border-radius:50%;background:var(--emerald);
box-shadow:0 0 12px var(--emerald);animation:pulse 2s infinite;}
.status-pill.off{background:rgba(251,113,133,0.08);color:var(--rose);}
.status-pill.off .dot{background:var(--rose);box-shadow:0 0 12px var(--rose);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.main{position:relative;z-index:1;padding:24px 32px;max-width:1200px;margin:0 auto;}
.section{margin-bottom:24px;}
.section-title{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim);
margin-bottom:12px;font-weight:600;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;
backdrop-filter:blur(10px);transition:border-color .3s,box-shadow .3s;}
.card:hover{border-color:var(--border2);box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.card-h{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;
border-bottom:1px solid var(--border);}
.card-h h3{font-size:14px;font-weight:700;}
.card-b{padding:20px;}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.b-green{background:rgba(52,211,153,0.12);color:var(--emerald);}
.b-blue{background:rgba(96,165,250,0.12);color:var(--blue);}
.b-violet{background:rgba(167,139,250,0.12);color:var(--violet);}
.b-amber{background:rgba(251,191,36,0.1);color:var(--amber);}
.b-rose{background:rgba(251,113,133,0.1);color:var(--rose);}
/* Toggle switch */
.toggle-row{display:flex;align-items:center;justify-content:space-between;padding:10px 0;
border-bottom:1px solid var(--border);}
.toggle-row:last-child{border-bottom:none;}
.toggle-label{font-size:13px;font-weight:500;}
.toggle-sub{font-size:11px;color:var(--dim);margin-top:2px;}
.toggle{position:relative;width:44px;height:24px;flex-shrink:0;}
.toggle input{opacity:0;width:0;height:0;}
.toggle .slider{position:absolute;cursor:pointer;inset:0;background:rgba(255,255,255,0.1);
border-radius:24px;transition:.3s;}
.toggle .slider::before{content:'';position:absolute;width:18px;height:18px;left:3px;bottom:3px;
background:#fff;border-radius:50%;transition:.3s;}
.toggle input:checked+.slider{background:var(--emerald);}
.toggle input:checked+.slider::before{transform:translateX(20px);}
/* Buttons */
.btn{padding:8px 16px;border-radius:10px;font-size:12px;font-weight:600;cursor:pointer;
border:1px solid var(--border);background:var(--glass);color:var(--text);transition:all .2s;
font-family:'Inter',sans-serif;}
.btn:hover{border-color:var(--border2);background:rgba(255,255,255,0.06);}
.btn:active{transform:scale(0.97);}
.btn-primary{background:rgba(102,126,234,0.2);border-color:rgba(102,126,234,0.3);color:#667eea;}
.btn-primary:hover{background:rgba(102,126,234,0.3);}
.btn-danger{background:rgba(251,113,133,0.1);border-color:rgba(251,113,133,0.2);color:var(--rose);}
.btn-danger:hover{background:rgba(251,113,133,0.2);}
.btn-group{display:flex;gap:8px;margin-top:12px;}
/* Quality selector */
.quality-sel{display:flex;gap:4px;background:rgba(255,255,255,0.04);border-radius:10px;padding:3px;}
.quality-opt{padding:6px 14px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;
color:var(--dim);transition:all .2s;border:none;background:none;font-family:'Inter',sans-serif;}
.quality-opt.active{background:rgba(102,126,234,0.2);color:#667eea;}
.quality-opt:hover{color:var(--text);}
/* Weight bars */
.weight-row{display:flex;align-items:center;gap:12px;padding:6px 0;}
.weight-name{font-size:12px;color:var(--dim);min-width:80px;}
.weight-bar{flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;}
.weight-fill{height:100%;border-radius:4px;transition:width .6s cubic-bezier(.16,1,.3,1);}
.weight-val{font-size:12px;font-weight:700;min-width:40px;text-align:right;font-feature-settings:'tnum';}
/* Repo list */
.repo-item{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--border);}
.repo-item:last-child{border-bottom:none;}
.repo-icon{font-size:20px;}
.repo-info{flex:1;min-width:0;}
.repo-path{font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.repo-meta{font-size:11px;color:var(--dim);margin-top:2px;}
/* Log viewer */
.log-viewer{background:#08090d;border:1px solid var(--border);border-radius:12px;
font-family:'JetBrains Mono',monospace;font-size:11px;padding:16px;max-height:400px;
overflow-y:auto;line-height:1.6;}
.log-line{color:var(--dim);}
.log-line .ts{color:var(--dim);}
.log-line .lvl-INFO{color:var(--blue);}
.log-line .lvl-WARNING{color:var(--amber);}
.log-line .lvl-ERROR{color:var(--rose);}
/* Toast */
.toast{position:fixed;bottom:24px;right:24px;padding:12px 20px;border-radius:12px;
background:rgba(14,17,24,0.95);border:1px solid var(--border2);backdrop-filter:blur(20px);
font-size:13px;font-weight:500;transform:translateY(80px);opacity:0;transition:all .4s cubic-bezier(.16,1,.3,1);z-index:999;}
.toast.show{transform:translateY(0);opacity:1;}
.toast.ok{border-color:rgba(52,211,153,0.3);color:var(--emerald);}
.toast.err{border-color:rgba(251,113,133,0.3);color:var(--rose);}
@keyframes flash{0%{background:rgba(102,126,234,0.15)}100%{background:transparent}}
.flash{animation:flash 1.5s ease-out;}
/* Federation warning */
.fed-warn{padding:12px 16px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);
border-radius:10px;font-size:12px;color:var(--amber);line-height:1.5;margin-bottom:12px;}
@media(max-width:900px){.grid,.grid3{grid-template-columns:1fr;}.main{padding:16px;}}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><h1>&#9889; Entroly</h1></div>
  <div class="nav-links">
    <a href="/">Dashboard</a>
    <a href="/controls" class="active">Controls</a>
  </div>
  <div class="status-pill" id="statusPill"><div class="dot"></div><span id="statusText">Connecting...</span></div>
</div>
<div class="main">

<!-- System -->
<div class="section">
<div class="section-title">System</div>
<div class="grid3">
  <div class="card"><div class="card-h"><h3>&#9881;&#65039; Daemon</h3><span class="badge b-green" id="daemonBadge">--</span></div>
  <div class="card-b">
    <div id="daemonInfo" style="font-size:13px;color:var(--dim);margin-bottom:12px;">Loading...</div>
    <div class="btn-group">
      <button class="btn btn-danger" onclick="ctrlPost('/api/control/stop')">Stop Daemon</button>
    </div>
  </div></div>

  <div class="card"><div class="card-h"><h3>&#128640; Proxy</h3><span class="badge b-blue" id="proxyBadge">--</span></div>
  <div class="card-b">
    <div class="toggle-row"><div><div class="toggle-label">Optimization</div>
    <div class="toggle-sub">Enable context optimization on all requests</div></div>
    <label class="toggle"><input type="checkbox" id="optToggle" checked onchange="toggleOpt(this)"><span class="slider"></span></label></div>
    <div class="toggle-row"><div><div class="toggle-label">Bypass Mode</div>
    <div class="toggle-sub">Forward requests raw without optimization</div></div>
    <label class="toggle"><input type="checkbox" id="bypassToggle" onchange="toggleBypass(this)"><span class="slider"></span></label></div>
  </div></div>

  <div class="card"><div class="card-h"><h3>&#127919; Quality Mode</h3><span class="badge b-violet" id="qualBadge">balanced</span></div>
  <div class="card-b">
    <div class="quality-sel" id="qualSel">
      <button class="quality-opt" data-q="fast" onclick="setQuality('fast')">&#9889; Fast</button>
      <button class="quality-opt active" data-q="balanced" onclick="setQuality('balanced')">&#9878;&#65039; Balanced</button>
      <button class="quality-opt" data-q="max" onclick="setQuality('max')">&#128142; Max</button>
    </div>
    <div style="font-size:11px;color:var(--dim);margin-top:12px;line-height:1.5;">
      <b>Fast:</b> Minimal context, lowest latency<br>
      <b>Balanced:</b> Optimal context/speed trade-off<br>
      <b>Max:</b> Maximum context quality, higher latency
    </div>
  </div></div>
</div></div>

<!-- Repos -->
<div class="section">
<div class="section-title">Repositories</div>
<div class="card"><div class="card-h"><h3>&#128193; Watched Repos</h3>
<button class="btn btn-primary" onclick="ctrlPost('/api/control/repos/reindex')">&#128260; Re-index All</button></div>
<div class="card-b" id="repoList"><div style="color:var(--dim);font-size:13px;">Loading...</div></div>
</div></div>

<!-- Learning -->
<div class="section">
<div class="section-title">Learning &amp; Intelligence</div>
<div class="grid">
  <div class="card"><div class="card-h"><h3>&#129504; PRISM Weights</h3><span class="badge b-violet" id="prismBadge">Default</span></div>
  <div class="card-b">
    <div id="weightsPanel">Loading...</div>
    <div class="btn-group">
      <button class="btn btn-primary" onclick="runAutotune()">&#9889; Run Autotune</button>
      <button class="btn btn-danger" onclick="resetWeights()">Reset Weights</button>
    </div>
  </div></div>

  <div class="card"><div class="card-h"><h3>&#128218; Epistemic Vault</h3><span class="badge b-blue" id="vaultBadge">--</span></div>
  <div class="card-b">
    <div class="toggle-row"><div><div class="toggle-label">Local Learning</div>
    <div class="toggle-sub">Adapt weights from proxy feedback</div></div>
    <label class="toggle"><input type="checkbox" id="learnToggle" checked onchange="toggleLearn(this)"><span class="slider"></span></label></div>
    <div id="vaultInfo" style="font-size:12px;color:var(--dim);margin-top:8px;">Loading...</div>
  </div></div>
</div></div>

<!-- Federation -->
<div class="section">
<div class="section-title">Federation</div>
<div class="card"><div class="card-h"><h3>&#127760; Global Learning Network</h3><span class="badge b-amber" id="fedBadge">OFF</span></div>
<div class="card-b">
  <div class="fed-warn">&#128274; Federation is <b>off by default</b>. No code, prompts, or symbols are ever uploaded.
  When enabled, only aggregate weight/strategy statistics are shared anonymously.</div>
  <div class="toggle-row"><div><div class="toggle-label">Enable Federation</div>
  <div class="toggle-sub">Opt-in to anonymous global learning</div></div>
  <label class="toggle"><input type="checkbox" id="fedToggle" onchange="toggleFed(this)"><span class="slider"></span></label></div>
</div></div></div>

<!-- Context -->
<div class="section">
<div class="section-title">Last Injected Context</div>
<div class="card"><div class="card-h"><h3>&#128269; Context Inspector</h3><span class="badge b-cyan" id="ctxBadge">--</span></div>
<div class="card-b" id="contextPanel"><div style="color:var(--dim);font-size:13px;">Run a proxy request to see context here</div></div>
</div></div>

<!-- Logs -->
<div class="section">
<div class="section-title">Logs</div>
<div class="card"><div class="card-h"><h3>&#128220; Live Logs</h3>
<button class="btn" onclick="refreshLogs()">&#128260; Refresh</button></div>
<div class="card-b"><div class="log-viewer" id="logViewer"><span class="log-line">Waiting for logs...</span></div></div>
</div></div>

</div>
<div class="toast" id="toast"></div>
<script>
function toast(msg,ok=true){const t=document.getElementById('toast');t.textContent=msg;
t.className='toast show '+(ok?'ok':'err');setTimeout(()=>t.className='toast',2500);}

async function ctrlPost(url,body={},msg='Done'){
  try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();if(d.ok)toast(msg);else toast(d.error||'Failed',false);refresh();}
  catch(e){toast('Connection error',false);}}

function toggleOpt(el){ctrlPost(el.checked?'/api/control/optimization/enable':'/api/control/optimization/pause',{},el.checked?'Optimization enabled':'Optimization paused');}
function toggleBypass(el){ctrlPost('/api/control/bypass',{enabled:el.checked},el.checked?'Bypass mode ON — requests forwarded raw':'Bypass mode OFF');}
function toggleLearn(el){ctrlPost('/api/control/learning/enable',{enabled:el.checked},el.checked?'Local learning enabled — weights will adapt from feedback':'Local learning paused — weights frozen');}
function toggleFed(el){if(el.checked)ctrlPost('/api/control/federation/enable',{},'Federation enabled');else ctrlPost('/api/control/federation/disable',{},'Federation disabled');}
function setQuality(m){document.querySelectorAll('.quality-opt').forEach(b=>b.classList.toggle('active',b.dataset.q===m));
ctrlPost('/api/control/quality',{mode:m},'Quality set to '+m);}
async function runAutotune(){toast('Running autotune...');await ctrlPost('/api/control/learning/autotune',{},'Autotune triggered — weights will update on next proxy request');document.getElementById('weightsPanel').classList.add('flash');setTimeout(()=>document.getElementById('weightsPanel').classList.remove('flash'),1600);}
async function resetWeights(){if(!confirm('Reset all learned weights to defaults?'))return;await ctrlPost('/api/control/learning/reset',{},'Weights reset to defaults (R=0.30 F=0.25 S=0.25 E=0.20)');document.getElementById('weightsPanel').classList.add('flash');setTimeout(()=>document.getElementById('weightsPanel').classList.remove('flash'),1600);}

function renderWeights(w){if(!w||!w.recency)return'<div style="color:var(--dim)">No weights</div>';
const colors=['#667eea','#f5576c','#4facfe','#43e97b'];
const names=['Recency','Frequency','Semantic','Entropy'];
const vals=[w.recency,w.frequency,w.semantic,w.entropy];
return vals.map((v,i)=>`<div class="weight-row"><span class="weight-name">${names[i]}</span>
<div class="weight-bar"><div class="weight-fill" style="width:${Math.min(v*200,100)}%;background:${colors[i]};"></div></div>
<span class="weight-val">${Math.round(v*100)}%</span></div>`).join('');}

function renderRepos(repos){if(!repos||!repos.length)return'<div style="color:var(--dim);font-size:13px;">No repos configured</div>';
return repos.map(r=>`<div class="repo-item"><span class="repo-icon">${r.watching?'&#128994;':'&#128308;'}</span>
<div class="repo-info"><div class="repo-path">${r.path}</div>
<div class="repo-meta">${r.indexed_files||0} files &middot; ${(r.total_tokens||0).toLocaleString()} tokens${r.last_sync?' &middot; synced '+new Date(r.last_sync*1000).toLocaleTimeString():''}</div></div>
<button class="btn" onclick="ctrlPost('/api/control/repos/reindex',{path:'${r.path.replace(/\\/g,'\\\\')}'})" style="flex-shrink:0;">Re-index</button></div>`).join('');}

function renderContext(ctx){if(!ctx||(!ctx.included&&!ctx.excluded))return'<div style="color:var(--dim);font-size:13px;">No context data yet</div>';
const inc=ctx.included||[];const exc=ctx.excluded||[];
let h='<div style="font-size:12px;color:var(--dim);margin-bottom:8px;">'+inc.length+' included &middot; '+exc.length+' excluded</div>';
inc.slice(0,8).forEach(f=>{const src=(f.source||f.id||'').split(/[\\/]/).pop();
const sc=f.scores||{}; const score=(sc.composite||0);
h+=`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid var(--border);">
<span style="color:var(--emerald);">&#10003; ${src}</span><span style="color:var(--dim);font-family:'JetBrains Mono',monospace;">${(score*100).toFixed(1)}%</span></div>`;});
return h;}

async function refreshLogs(){
  try{const r=await fetch('/api/control/logs');const d=await r.json();
  const el=document.getElementById('logViewer');
  if(d.lines&&d.lines.length>0){el.innerHTML=d.lines.map(l=>{
    let cls='';if(l.includes('ERROR'))cls='lvl-ERROR';else if(l.includes('WARNING'))cls='lvl-WARNING';
    else if(l.includes('INFO'))cls='lvl-INFO';
    return'<div class="log-line"><span class="'+cls+'">'+l.replace(/</g,'&lt;')+'</span></div>';}).join('');}
  else{el.innerHTML='<span class="log-line">No log entries yet</span>';}}catch(e){}}

async function refresh(){
  try{
    const sr=await fetch('/api/control/status');const s=await sr.json();
    const pill=document.getElementById('statusPill');const stxt=document.getElementById('statusText');
    if(s.error){pill.className='status-pill off';stxt.textContent='Daemon not running';return;}
    pill.className='status-pill';stxt.textContent='v'+s.version+' \u00b7 '+(s.uptime_s>0?Math.round(s.uptime_s)+'s uptime':'starting');
    document.getElementById('daemonBadge').textContent=s.status;
    document.getElementById('daemonBadge').className='badge '+(s.status==='running'?'b-green':'b-rose');
    document.getElementById('daemonInfo').innerHTML=
      'Status: <b>'+s.status+'</b><br>Uptime: '+(s.uptime_s>0?Math.round(s.uptime_s/60)+'m':'--');
    document.getElementById('proxyBadge').textContent=s.proxy.running?'Running':'Stopped';
    document.getElementById('proxyBadge').className='badge '+(s.proxy.running?'b-green':'b-rose');
    document.getElementById('optToggle').checked=s.optimization.enabled;
    document.getElementById('bypassToggle').checked=s.optimization.bypass;
    document.getElementById('qualBadge').textContent=s.optimization.quality;
    document.querySelectorAll('.quality-opt').forEach(b=>b.classList.toggle('active',b.dataset.q===s.optimization.quality));
    document.getElementById('fedToggle').checked=s.federation.enabled;
    document.getElementById('fedBadge').textContent=s.federation.enabled?s.federation.mode.toUpperCase():'OFF';
    document.getElementById('fedBadge').className='badge '+(s.federation.enabled?'b-green':'b-amber');
    // Enrich repo data with live engine stats via /api/metrics
    let repos=s.repos||[];
    try{const mr=await fetch('/api/metrics');const md=await mr.json();
      const ss=(md.stats||{}).session||{};
      if(repos.length>0&&(repos[0].indexed_files||0)===0&&ss.total_fragments>0){
        repos[0].indexed_files=ss.total_fragments;
        repos[0].total_tokens=ss.total_tokens_tracked||0;}
    }catch(e){}
    renderReposFromState(repos);
  }catch(e){document.getElementById('statusPill').className='status-pill off';
  document.getElementById('statusText').textContent='Disconnected';}

  try{const lr=await fetch('/api/control/learning');const l=await lr.json();
  document.getElementById('weightsPanel').innerHTML=renderWeights(l.weights);
  const w=l.weights||{};
  const isDefault=(w.recency===0.3&&w.frequency===0.25&&w.semantic===0.25&&w.entropy===0.2);
  document.getElementById('prismBadge').textContent=isDefault?'Default':'RL-Learned';
  document.getElementById('prismBadge').className='badge '+(isDefault?'b-amber':'b-violet');
  document.getElementById('learnToggle').checked=l.local_enabled;
  document.getElementById('vaultBadge').textContent=l.local_enabled?'Active':'Paused';
  document.getElementById('vaultBadge').className='badge '+(l.local_enabled?'b-green':'b-amber');
  document.getElementById('vaultInfo').innerHTML=w.recency?
    'R='+w.recency.toFixed(2)+' F='+w.frequency.toFixed(2)+' S='+w.semantic.toFixed(2)+' E='+w.entropy.toFixed(2):
    'No weight data';
  }catch(e){}

  try{const cr=await fetch('/api/control/context/last');const c=await cr.json();
  document.getElementById('contextPanel').innerHTML=renderContext(c);
  const n=(c.included||[]).length;
  document.getElementById('ctxBadge').textContent=n>0?n+' fragments':'--';
  }catch(e){}
}

function renderReposFromState(repos){document.getElementById('repoList').innerHTML=renderRepos(repos);}

refresh();setInterval(refresh,5000);setTimeout(refreshLogs,1000);setInterval(refreshLogs,10000);
</script>
</body>
</html>"""
