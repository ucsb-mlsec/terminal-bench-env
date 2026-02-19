FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Rust toolchain
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create workspace directory structure
RUN mkdir -p /workspace/api-client/src && \
    mkdir -p /workspace/data-processor/src && \
    mkdir -p /workspace/web-server/src && \
    mkdir -p /workspace/shared-models/src && \
    mkdir -p /workspace/cli-tool/src

# Copy all generated files to workspace
COPY api-client/Cargo.toml /workspace/api-client/Cargo.toml
COPY api-client/src/lib.rs /workspace/api-client/src/lib.rs

COPY data-processor/Cargo.toml /workspace/data-processor/Cargo.toml
COPY data-processor/src/lib.rs /workspace/data-processor/src/lib.rs

COPY web-server/Cargo.toml /workspace/web-server/Cargo.toml
COPY web-server/src/main.rs /workspace/web-server/src/main.rs

COPY shared-models/Cargo.toml /workspace/shared-models/Cargo.toml
COPY shared-models/src/lib.rs /workspace/shared-models/src/lib.rs

COPY cli-tool/Cargo.toml /workspace/cli-tool/Cargo.toml
COPY cli-tool/src/main.rs /workspace/cli-tool/src/main.rs

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
