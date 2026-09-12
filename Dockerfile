# Minimal single-platform image, kept so that a bare `docker build .` works.
#
# The published image is built from `deploy/Dockerfile.entroly`, which is what
# deploy/docker-compose.yml and .github/workflows/entroly-publish.yml use and what
# ships multi-platform (amd64 + arm64). Prefer it for anything but a quick
# local build:
#
#     docker build -f deploy/Dockerfile.entroly -t entroly .
#
# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the application and its dependencies
RUN pip install --no-cache-dir .[full]

# Set the entrypoint to run the MCP server
ENTRYPOINT ["entroly", "serve"]
