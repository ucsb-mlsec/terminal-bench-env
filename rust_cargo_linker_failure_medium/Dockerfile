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

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/rust-project/src && \
    mkdir -p /opt/custom/lib

COPY rust-project/Cargo.toml /workspace/rust-project/Cargo.toml
COPY rust-project/Cargo.lock /workspace/rust-project/Cargo.lock
COPY rust-project/src/main.rs /workspace/rust-project/src/main.rs
COPY opt/custom/lib/libssl.so.1.1 /opt/custom/lib/libssl.so.1.1
COPY opt/custom/lib/libcrypto.so.1.1 /opt/custom/lib/libcrypto.so.1.1

WORKDIR /workspace