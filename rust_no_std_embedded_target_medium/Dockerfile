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
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.75.0 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add thumbv7em-none-eabihf

RUN mkdir -p /workspace/blinky_app/src /workspace/blinky_app/.cargo

COPY blinky_app/Cargo.toml /workspace/blinky_app/Cargo.toml
COPY blinky_app/src/main.rs /workspace/blinky_app/src/main.rs
COPY blinky_app/.cargo/config.toml /workspace/blinky_app/.cargo/config.toml
COPY blinky_app/memory.x /workspace/blinky_app/memory.x

WORKDIR /workspace/blinky_app