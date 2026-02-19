FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install necessary dependencies for Rust FFI and bindgen
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    gcc \
    clang \
    libclang-dev \
    llvm-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y clang libclang-dev gcc make llvm && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install bindgen-cli
RUN /root/.cargo/bin/cargo install bindgen-cli

# Create necessary directories
RUN mkdir -p /workspace/mathlib

# Copy C library files
COPY mathlib/math_ops.h /workspace/mathlib/math_ops.h
COPY mathlib/math_ops.c /workspace/mathlib/math_ops.c
COPY mathlib/libmathops.a /workspace/mathlib/libmathops.a

WORKDIR /workspace