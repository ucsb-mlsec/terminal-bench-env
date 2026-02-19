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

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y rustc cargo musl-tools && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

ENV PATH="/root/.cargo/bin:${PATH}"

RUN /root/.cargo/bin/rustup target add x86_64-unknown-linux-musl

RUN mkdir -p /workspace/textproc/src /workspace/textproc/data

COPY textproc/Cargo.toml /workspace/textproc/Cargo.toml
COPY textproc/src/main.rs /workspace/textproc/src/main.rs
COPY data/sample.txt /workspace/textproc/sample.txt

WORKDIR /workspace