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
    gcc \
    pkg-config \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup default stable

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/signal_processor/csrc && \
    mkdir -p /workspace/signal_processor/src

COPY workspace/signal_processor/csrc/signal.h /workspace/signal_processor/csrc/signal.h
COPY workspace/signal_processor/csrc/filter.c /workspace/signal_processor/csrc/filter.c
COPY workspace/signal_processor/csrc/analysis.c /workspace/signal_processor/csrc/analysis.c
COPY workspace/signal_processor/src/lib.rs /workspace/signal_processor/src/lib.rs
COPY workspace/signal_processor/Cargo.toml /workspace/signal_processor/Cargo.toml

WORKDIR /workspace/signal_processor