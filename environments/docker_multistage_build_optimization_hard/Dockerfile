FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /workspace/dockerfiles /workspace/solution

# Copy all generated Dockerfile samples to the container
COPY dockerfiles/service-api.dockerfile /workspace/dockerfiles/service-api.dockerfile
COPY dockerfiles/service-worker.dockerfile /workspace/dockerfiles/service-worker.dockerfile
COPY dockerfiles/service-frontend.dockerfile /workspace/dockerfiles/service-frontend.dockerfile
COPY dockerfiles/service-legacy.dockerfile /workspace/dockerfiles/service-legacy.dockerfile
COPY dockerfiles/service-monolith.dockerfile /workspace/dockerfiles/service-monolith.dockerfile
COPY dockerfiles/service-database.dockerfile /workspace/dockerfiles/service-database.dockerfile
COPY dockerfiles/service-auth.dockerfile /workspace/dockerfiles/service-auth.dockerfile
COPY dockerfiles/service-analytics.dockerfile /workspace/dockerfiles/service-analytics.dockerfile
COPY dockerfiles/service-gateway.dockerfile /workspace/dockerfiles/service-gateway.dockerfile
COPY dockerfiles/service-notification.dockerfile /workspace/dockerfiles/service-notification.dockerfile

WORKDIR /workspace