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
