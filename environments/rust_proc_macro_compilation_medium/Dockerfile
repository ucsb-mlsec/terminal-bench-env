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
RUN apt-get update && apt-get install -y rustc cargo && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/derive_macro/src /workspace/app/src

COPY Cargo.toml /workspace/Cargo.toml
COPY derive_macro/Cargo.toml /workspace/derive_macro/Cargo.toml
COPY derive_macro/src/lib.rs /workspace/derive_macro/src/lib.rs
COPY app/Cargo.toml /workspace/app/Cargo.toml
COPY app/src/main.rs /workspace/app/src/main.rs

WORKDIR /workspace