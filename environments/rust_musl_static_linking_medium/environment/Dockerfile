FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    musl-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.70.0 && \
    . $HOME/.cargo/env && \
    rustup target add x86_64-unknown-linux-musl

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/rust-app/src

COPY rust-app/Cargo.toml /workspace/rust-app/Cargo.toml
COPY rust-app/src/main.rs /workspace/rust-app/src/main.rs

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
