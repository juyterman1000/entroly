# Example: Integrate Entroly with MCP

from entroly import load_secrets
from mcp import MCPServer

# Load secrets using Entroly
env_secrets = load_secrets()

# Pass secrets as context to MCP server
server = MCPServer(context=env_secrets)

# Start the MCP server
server.run()

# This example assumes you have configured Entroly to fetch secrets from your backend (file, cloud, etc.)
# and MCPServer accepts a context dictionary for environment variables/secrets.
