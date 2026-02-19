FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install fuzzing and development tools
RUN apt-get update && \
    apt-get install -y \
    clang \
    llvm \
    build-essential \
    afl++ \
    libasan6 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc clang make && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /opt/validator /tmp

# Copy validator library files
COPY opt/validator/validator.c /opt/validator/validator.c
COPY opt/validator/validator.h /opt/validator/validator.h
COPY opt/validator/README.md /opt/validator/README.md

# Set working directory
WORKDIR /tmp