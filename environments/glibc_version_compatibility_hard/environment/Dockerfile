FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install necessary tools for binary analysis
RUN apt-get update && \
    apt-get install -y \
    binutils \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y binutils && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/app /solution

# Copy the binary to the container
COPY app/legacy_service /workspace/app/legacy_service

# Make the binary executable
RUN chmod +x /workspace/app/legacy_service

# Set working directory
WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
