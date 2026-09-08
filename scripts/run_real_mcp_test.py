import os
import json
from pathlib import Path

# Setup mock environment so server knows where to write
os.environ["ENTROLY_DIR"] = str(Path.home() / ".entroly")

from entroly.server import create_mcp_server

def run_real_mcp_test():
    print("Initializing Real MCP Server (as Claude Desktop would)...")
    mcp = create_mcp_server()

    print("\nSending Real Code-Inspection request to Entroly MCP...")
    # This invokes the RAVS-wrapped MCP tool directly
    try:
        res = mcp._tools["optimize_context"](
            task="Find all CLI arguments in entroly/cli.py and summarize the commands",
            budget=1024,
            selector="auto",
            exclude=None
        )
        print(f"Success! Optimized down to {len(res)} characters.")
    except Exception as e:
        print(f"Tool error: {e}")

    print("\nSending Real Computation request to Entroly MCP...")
    try:
        res2 = mcp._tools["optimize_context"](
            task="We need to verify the math logic for Token Pricing. What is 256 tokens * $0.00001?",
            budget=512,
            selector="auto",
            exclude=None
        )
        print(f"Success! Optimized down to {len(res2)} characters.")
    except Exception as e:
        print(f"Tool error: {e}")

if __name__ == "__main__":
    run_real_mcp_test()
