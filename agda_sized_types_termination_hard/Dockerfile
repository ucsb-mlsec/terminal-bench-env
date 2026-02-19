FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Agda and dependencies
RUN apt-get update && \
    apt-get install -y agda libghc-agda-dev && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/stream-verification

# Copy Agda source files to the project location
COPY stream-verification/Stream.agda /workspace/stream-verification/Stream.agda
COPY stream-verification/CoList.agda /workspace/stream-verification/CoList.agda
COPY stream-verification/Nat.agda /workspace/stream-verification/Nat.agda

# Set working directory to project location
WORKDIR /workspace/stream-verification