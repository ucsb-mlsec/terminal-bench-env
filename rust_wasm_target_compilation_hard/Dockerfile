FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Rust and dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Rust 1.75.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.75.0 && \
    . $HOME/.cargo/env && \
    rustup target add wasm32-unknown-unknown

# Set PATH for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Create project directory structure
RUN mkdir -p /workspace/calculator-wasm/src

# Copy project files
COPY calculator-wasm/Cargo.toml /workspace/calculator-wasm/Cargo.toml
COPY calculator-wasm/src/lib.rs /workspace/calculator-wasm/src/lib.rs

WORKDIR /workspace/calculator-wasm