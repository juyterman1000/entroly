#!/usr/bin/env python3
"""
Entroly Demo Video Generator
=============================

Generates a stunning animated terminal recording as SVG, GIF, or HTML.
This creates the hero visual for the README — showing Entroly's value
proposition in a compelling 25-second animated terminal session.

Usage:
    python docs/generate_demo.py              # Generate SVG (default)
    python docs/generate_demo.py --format gif  # Generate GIF (requires Pillow)
    python docs/generate_demo.py --format html # Generate interactive HTML

The output is saved to docs/assets/demo_animated.svg (or .gif/.html).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ── Terminal Theme (Dracula) ──
COLORS = {
    "bg": "#282a36",
    "bg_dark": "#1e1f29",
    "fg": "#f8f8f2",
    "green": "#50fa7b",
    "cyan": "#8be9fd",
    "yellow": "#f1fa8c",
    "red": "#ff5555",
    "purple": "#bd93f9",
    "orange": "#ffb86c",
    "gray": "#6272a4",
    "selection": "#44475a",
    "bright": "#ffffff",
}

# ── Storyboard ──
# Each frame is (delay_ms, lines) where lines are (text, color, bold, indent)
FRAMES = [
    # Frame 0: Command prompt
    (800, [
        ("$ entroly demo", "bright", True, 0),
    ]),
    # Frame 1: Indexing spinner
    (1200, [
        ("", None, False, 0),
        ("  \u23f3 Indexing your codebase...", "gray", False, 0),
    ]),
    # Frame 2: Indexed result
    (1000, [
        ("  \u2713 Indexed 847 files (186,420 tokens) in 1.8s", "green", False, 0),
    ]),
    # Frame 3: WITHOUT box header
    (1200, [
        ("", None, False, 0),
        ("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510", "red", False, 0),
        ("  \u2502  WITHOUT ENTROLY                                       \u2502", "red", True, 0),
        ("  \u2502  Your AI sees: 5-10 files (rest is invisible)          \u2502", "fg", False, 0),
        ("  \u2502  Tokens sent:  186,420 (entire raw dump)               \u2502", "fg", False, 0),
        ("  \u2502  Cost / 1K requests: ~$560                             \u2502", "red", True, 0),
        ("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518", "red", False, 0),
    ]),
    # Frame 4: WITH box header
    (1200, [
        ("", None, False, 0),
        ("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510", "green", False, 0),
        ("  \u2502  WITH ENTROLY                                          \u2502", "green", True, 0),
        ("  \u2502  Your AI sees: ALL 847 files (variable resolution)     \u2502", "fg", False, 0),
    ]),
    # Frame 5: Query 1
    (900, [
        ("  \u2502                                                        \u2502", "green", False, 0),
        ("  \u2502  Q1: \"Fix the authentication bug\"                      \u2502", "cyan", False, 0),
    ]),
    (700, [
        ("  \u2502      \u2192 12 fragments, 7,450 tokens       \u25bc 96% saved  \u2502", "green", True, 0),
    ]),
    # Frame 6: Query 2
    (900, [
        ("  \u2502                                                        \u2502", "green", False, 0),
        ("  \u2502  Q2: \"Explain the module architecture\"                 \u2502", "cyan", False, 0),
    ]),
    (700, [
        ("  \u2502      \u2192 35 fragments, 10,240 tokens      \u25bc 94% saved  \u2502", "green", True, 0),
    ]),
    # Frame 7: Query 3
    (900, [
        ("  \u2502                                                        \u2502", "green", False, 0),
        ("  \u2502  Q3: \"Find SQL injection vulnerabilities\"              \u2502", "cyan", False, 0),
    ]),
    (700, [
        ("  \u2502      \u2192 42 fragments, 9,320 tokens       \u25bc 95% saved  \u2502", "green", True, 0),
    ]),
    # Frame 8: Summary
    (600, [
        ("  \u2502                                                        \u2502", "green", False, 0),
        ("  \u2502  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500  \u2502", "green", False, 0),
    ]),
    (800, [
        ("  \u2502  Average:   70-95% fewer tokens                        \u2502", "green", True, 0),
        ("  \u2502  Cost:      ~$28 / 1K requests (was $560)              \u2502", "green", True, 0),
        ("  \u2502  Pipeline:  8.2ms (entropy \u2192 knapsack \u2192 PRISM)         \u2502", "purple", False, 0),
    ]),
    (400, [
        ("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518", "green", False, 0),
    ]),
    # Frame 9: Final message
    (1200, [
        ("", None, False, 0),
        ("  \u2713 Your AI sees more. You pay less. Zero config changes.", "bright", True, 0),
    ]),
    (1500, [
        ("", None, False, 0),
        ("  Start now:  pip install entroly && entroly proxy", "cyan", False, 0),
    ]),
]

LINE_HEIGHT = 18
FONT_SIZE = 13
PADDING_X = 36
PADDING_Y = 60  # below title bar
TERMINAL_WIDTH = 840
TITLE_BAR_HEIGHT = 36
CHAR_WIDTH = 7.8  # approximate monospace char width


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_svg() -> str:
    """Generate an animated SVG terminal recording with SMIL animations."""

    # Calculate total lines and timing
    all_lines: list[tuple[str, str, bool, float]] = []  # text, color, bold, appear_time_s
    t = 0.0
    for delay_ms, lines in FRAMES:
        t += delay_ms / 1000.0
        for text, color, bold, indent in lines:
            all_lines.append((text, color or "fg", bold, t))
            t += 0.08  # stagger within frame

    total_lines = len(all_lines)
    terminal_height = PADDING_Y + (total_lines * LINE_HEIGHT) + 40
    total_duration = t + 4.0  # hold final frame

    parts = []

    # SVG header
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {TERMINAL_WIDTH} {terminal_height}" width="{TERMINAL_WIDTH}" height="{terminal_height}">')
    parts.append("""  <defs>
    <style>
      .t { font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', 'Menlo', monospace; font-size: 13px; }
      .b { font-weight: 700; }
    </style>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#282a36"/>
      <stop offset="100%" stop-color="#1e1f29"/>
    </linearGradient>
    <linearGradient id="tb" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#44475a"/>
      <stop offset="100%" stop-color="#383a4a"/>
    </linearGradient>
  </defs>""")

    # Background
    parts.append(f'  <rect width="{TERMINAL_WIDTH}" height="{terminal_height}" rx="0" fill="#0d0e17"/>')

    # Terminal window
    parts.append(f'  <rect x="20" y="12" width="{TERMINAL_WIDTH - 40}" height="{terminal_height - 24}" rx="10" fill="url(#bg)" stroke="#44475a" stroke-width="1"/>')

    # Title bar
    parts.append(f'  <rect x="20" y="12" width="{TERMINAL_WIDTH - 40}" height="{TITLE_BAR_HEIGHT}" rx="10" fill="url(#tb)"/>')
    parts.append(f'  <rect x="20" y="38" width="{TERMINAL_WIDTH - 40}" height="10" fill="url(#tb)"/>')

    # Traffic lights
    parts.append('  <circle cx="44" cy="30" r="6" fill="#ff5f56"/>')
    parts.append('  <circle cx="64" cy="30" r="6" fill="#ffbd2e"/>')
    parts.append('  <circle cx="84" cy="30" r="6" fill="#27c93f"/>')

    # Title text
    parts.append(f'  <text x="{TERMINAL_WIDTH // 2}" y="34" text-anchor="middle" class="t" fill="{COLORS["gray"]}" font-size="12">entroly \u2014 Developer Demo</text>')

    # Animated text lines
    for i, (text, color, bold, appear_time) in enumerate(all_lines):
        if not text:
            continue

        y = PADDING_Y + (i * LINE_HEIGHT) + 14
        color_hex = COLORS.get(color, COLORS["fg"])
        bold_class = " b" if bold else ""
        escaped = _escape_xml(text)

        parts.append('  <g opacity="0">')
        parts.append(f'    <animate attributeName="opacity" from="0" to="1" begin="{appear_time:.2f}s" dur="0.25s" fill="freeze"/>')
        parts.append(f'    <text x="{PADDING_X}" y="{y}" class="t{bold_class}" fill="{color_hex}">{escaped}</text>')
        parts.append('  </g>')

    # Blinking cursor at the end
    last_y = PADDING_Y + (total_lines * LINE_HEIGHT) + 2
    cursor_appear = all_lines[-1][3] + 0.5 if all_lines else 1.0
    parts.append(f'  <rect x="{PADDING_X}" y="{last_y}" width="8" height="16" fill="{COLORS["fg"]}" opacity="0">')
    parts.append(f'    <animate attributeName="opacity" from="0" to="1" begin="{cursor_appear:.2f}s" dur="0.01s" fill="freeze"/>')
    parts.append(f'    <animate attributeName="opacity" values="1;1;0;0" dur="1s" begin="{cursor_appear:.2f}s" repeatCount="indefinite"/>')
    parts.append('  </rect>')

    parts.append('</svg>')
    return '\n'.join(parts)


def generate_html() -> str:
    """Generate an interactive HTML page with the terminal animation."""

    # Collect all lines with timing
    all_entries: list[tuple[str, str, bool, int]] = []  # text, color, bold, delay_total_ms
    t = 0
    for delay_ms, lines in FRAMES:
        t += delay_ms
        for text, color, bold, indent in lines:
            all_entries.append((text, color or "fg", bold, t))
            t += 60

    lines_json = []
    for text, color, bold, delay in all_entries:
        color_hex = COLORS.get(color, COLORS["fg"])
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines_json.append(f'    {{t:"{escaped}",c:"{color_hex}",b:{str(bold).lower()},d:{delay}}}')

    lines_str = ",\n".join(lines_json)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entroly \u2014 Developer Demo</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d0e17;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 20px;
  }}
  .terminal {{
    background: linear-gradient(180deg, #282a36 0%, #1e1f29 100%);
    border: 1px solid #44475a;
    border-radius: 10px;
    width: 800px;
    max-width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    overflow: hidden;
  }}
  .title-bar {{
    background: linear-gradient(180deg, #44475a 0%, #383a4a 100%);
    height: 36px;
    display: flex;
    align-items: center;
    padding: 0 14px;
    position: relative;
  }}
  .traffic-light {{
    width: 12px; height: 12px; border-radius: 50%;
    margin-right: 8px; display: inline-block;
  }}
  .tl-red {{ background: #ff5f56; }}
  .tl-yellow {{ background: #ffbd2e; }}
  .tl-green {{ background: #27c93f; }}
  .title-text {{
    position: absolute; left: 50%; transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #6272a4;
  }}
  .content {{
    padding: 16px 20px 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.5;
    min-height: 480px;
  }}
  .line {{
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.25s ease, transform 0.25s ease;
    white-space: pre;
  }}
  .line.show {{
    opacity: 1;
    transform: translateY(0);
  }}
  .cursor {{
    display: inline-block;
    width: 8px; height: 16px;
    background: #f8f8f2;
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
    opacity: 0;
  }}
  .cursor.show {{ opacity: 1; }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}
  .controls {{
    display: flex;
    justify-content: center;
    gap: 12px;
    padding: 12px;
    border-top: 1px solid #44475a;
  }}
  .controls button {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: #44475a;
    color: #f8f8f2;
    border: 1px solid #6272a4;
    border-radius: 4px;
    padding: 6px 16px;
    cursor: pointer;
    transition: background 0.2s;
  }}
  .controls button:hover {{ background: #6272a4; }}
</style>
</head>
<body>
<div class="terminal">
  <div class="title-bar">
    <span class="traffic-light tl-red"></span>
    <span class="traffic-light tl-yellow"></span>
    <span class="traffic-light tl-green"></span>
    <span class="title-text">entroly \u2014 Developer Demo</span>
  </div>
  <div class="content" id="content"></div>
  <div class="controls">
    <button onclick="replay()">Replay</button>
    <button onclick="skipToEnd()">Skip to End</button>
  </div>
</div>
<script>
const L=[
{lines_str}
];
const content=document.getElementById('content');
let timeouts=[];
function clear(){{ timeouts.forEach(clearTimeout); timeouts=[]; content.innerHTML=''; }}
function play() {{
  clear();
  L.forEach((l,i) => {{
    const div=document.createElement('div');
    div.className='line';
    if(l.b) div.style.fontWeight='700';
    div.style.color=l.c;
    div.textContent=l.t||'\\u00a0';
    content.appendChild(div);
    timeouts.push(setTimeout(()=>div.classList.add('show'), l.d));
  }});
  // Add cursor
  const cur=document.createElement('span');
  cur.className='cursor';
  content.appendChild(cur);
  const lastDelay=L[L.length-1].d+500;
  timeouts.push(setTimeout(()=>cur.classList.add('show'), lastDelay));
}}
function replay() {{ play(); }}
function skipToEnd() {{
  clear();
  L.forEach(l => {{
    const div=document.createElement('div');
    div.className='line show';
    if(l.b) div.style.fontWeight='700';
    div.style.color=l.c;
    div.textContent=l.t||'\\u00a0';
    content.appendChild(div);
  }});
  const cur=document.createElement('span');
  cur.className='cursor show';
  content.appendChild(cur);
}}
play();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate Entroly demo video/animation")
    parser.add_argument("--format", choices=["svg", "html", "both"], default="both",
                        help="Output format (default: both)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: docs/assets/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("svg", "both"):
        svg_path = output_dir / "demo_animated.svg"
        svg_path.write_text(generate_svg(), encoding="utf-8")
        print(f"  \033[38;5;82m\u2713\033[0m SVG saved: {svg_path}")

    if args.format in ("html", "both"):
        html_path = output_dir / "demo.html"
        html_path.write_text(generate_html(), encoding="utf-8")
        print(f"  \033[38;5;82m\u2713\033[0m HTML saved: {html_path}")

    print("\n  \033[38;5;45mTip:\033[0m Open the HTML file in a browser for the interactive version!")
    print("  \033[38;5;45mTip:\033[0m Use the SVG in your README for GitHub rendering.")


if __name__ == "__main__":
    main()
