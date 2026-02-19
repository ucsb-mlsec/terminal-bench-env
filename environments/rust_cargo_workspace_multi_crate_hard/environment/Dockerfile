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
    apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/core/src /workspace/storage/src /workspace/api/src

COPY Cargo.toml /workspace/Cargo.toml
COPY core/Cargo.toml /workspace/core/Cargo.toml
COPY core/src/lib.rs /workspace/core/src/lib.rs
COPY storage/Cargo.toml /workspace/storage/Cargo.toml
COPY storage/src/lib.rs /workspace/storage/src/lib.rs
COPY api/Cargo.toml /workspace/api/Cargo.toml
COPY api/src/lib.rs /workspace/api/src/lib.rs

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
