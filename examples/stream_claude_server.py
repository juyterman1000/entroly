import asyncio
import sys
import os
from aiohttp import web

html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Entroly + Claude Code Real-Time Execution</title>
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
            font-size: 14px;
            line-height: 1.5;
            padding: 20px;
            margin: 0;
            overflow-y: auto;
        }
        #term {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div id="term"></div>
    <script>
        const term = document.getElementById('term');
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(proto + '//' + window.location.host + '/ws');

        ws.onopen = function() {
            term.innerHTML += "<span style='color: #58a6ff'>workspace/langfuse></span> entroly wrap claude -p \\"Explain how trace ingestion works in the worker package\\"\\n\\n";
            ws.send("start");
        };

        // Model output is untrusted: a prompt-injected reply can contain
        // markup. Escape it first, then turn the ANSI colour codes (which
        // escaping leaves alone) into the only spans this page creates.
        function escapeHtml(s) {
            return s.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
        }

        ws.onmessage = function(event) {
            let text = escapeHtml(event.data);
            // Basic ANSI color parsing
            text = text.replace(/\\x1b\\[32m/g, "<span style='color: #3fb950'>"); // green
            text = text.replace(/\\x1b\\[31m/g, "<span style='color: #f85149'>"); // red
            text = text.replace(/\\x1b\\[33m/g, "<span style='color: #d29922'>"); // yellow
            text = text.replace(/\\x1b\\[34m/g, "<span style='color: #58a6ff'>"); // blue
            text = text.replace(/\\x1b\\[35m/g, "<span style='color: #bc8cff'>"); // magenta
            text = text.replace(/\\x1b\\[36m/g, "<span style='color: #39c5cf'>"); // cyan
            text = text.replace(/\\x1b\\[1m/g, "<span style='font-weight: bold; color: #f0f6fc'>"); // bold
            text = text.replace(/\\x1b\\[2m/g, "<span style='color: #8b949e'>"); // dim
            text = text.replace(/\\x1b\\[0m/g, "</span>"); // reset

            // Fix newlines that might be missing from terminal stream
            text = text.replace(/\\r/g, "");

            term.innerHTML += text;
            window.scrollTo(0, document.body.scrollHeight);
        };

        ws.onclose = function() {
            term.innerHTML += "\\n\\n<span style='color: #8b949e'>[Process completed]</span>";
        };
    </script>
</body>
</html>
"""


async def handle_index(request):
    return web.Response(text=html_page, content_type="text/html")


HOST = "127.0.0.1"
PORT = 8083
# Browsers do not apply CORS to WebSockets: any page the user visits can open
# ws://localhost:8083/ws. Without these checks such a page could start a Claude
# run over the user's repository and read every line of its output.
ALLOWED_ORIGINS = frozenset({f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"})
ALLOWED_HOSTS = frozenset({f"localhost:{PORT}", f"127.0.0.1:{PORT}"})


def request_is_from_this_page(origin: str | None, host: str | None) -> bool:
    """True only for this server's own page, addressed by a loopback name.

    The Host check also stops DNS rebinding, where an attacker's domain is
    re-pointed at 127.0.0.1 so its page looks same-origin to itself.
    """
    return origin in ALLOWED_ORIGINS and host in ALLOWED_HOSTS


async def handle_websocket(request):
    if not request_is_from_this_page(request.headers.get("Origin"), request.headers.get("Host")):
        return web.Response(status=403, text="cross-site WebSocket connections are refused")
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.data == "start":
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Force color output for CLI tools if possible
            env["FORCE_COLOR"] = "1"
            env["CLICOLOR_FORCE"] = "1"

            # Use entroly wrap claude inside the langfuse directory
            process = await asyncio.create_subprocess_shell(
                'entroly wrap claude -p "Explain how trace ingestion works in the worker package"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=os.environ.get("LANGFUSE_ROOT", "."),
            )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                await ws.send_str(line.decode("utf-8", errors="replace"))
                await asyncio.sleep(0.01)

            await process.wait()
            await ws.close()
            break

    return ws


app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_get("/ws", handle_websocket)

if __name__ == "__main__":
    print(f"Starting Claude live streaming server on http://localhost:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
