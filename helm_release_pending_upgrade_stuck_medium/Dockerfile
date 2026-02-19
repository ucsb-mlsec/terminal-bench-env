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
RUN mkdir -p /workspace/helm_metadata

# Copy Helm metadata files to the expected locations
COPY data/helm_metadata/release.v1.json /workspace/helm_metadata/release.v1.json
COPY data/helm_metadata/release.v2.json /workspace/helm_metadata/release.v2.json
COPY data/helm_metadata/current_state.json /workspace/helm_metadata/current_state.json

# Set working directory
WORKDIR /workspace