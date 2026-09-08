import time
import subprocess
import httpx
import os

# 1. Start proxy in background
print("Starting Entroly proxy...")
proxy = subprocess.Popen(
    ["python", "-m", "entroly.cli", "proxy", "--port", "9388"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(3) # Wait for it to boot

messages_list = [
    [{"role": "user", "content": "Calculate 253 * 18"}],
    [{"role": "user", "content": "What are the classes defined in entroly/cli.py?"}],
    [{"role": "user", "content": "Write a python script that prints hello world."}],
]

headers = {
    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy-key')}",
    "Content-Type": "application/json"
}

print("Sending REAL requests to proxy...")
for idx, msgs in enumerate(messages_list):
    print(f"Request {idx+1}: {msgs[0]['content']}")
    try:
        resp = httpx.post(
            "http://localhost:9388/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": msgs,
                "max_tokens": 50
            },
            headers=headers,
            timeout=10.0
        )
        print("  -> Status:", resp.status_code)
        if resp.status_code == 200:
            print("  -> Reply:", resp.json()["choices"][0]["message"]["content"][:50].replace("\n", " ") + "...")
    except Exception as e:
        print("  -> Error:", str(e))

# Shut down proxy
proxy.terminate()
print("Proxy terminated.")
