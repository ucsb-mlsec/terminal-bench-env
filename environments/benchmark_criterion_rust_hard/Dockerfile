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
RUN apt-get update && apt-get install -y cargo rustc && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /workspace/math_utils/src && \
    mkdir -p /workspace/math_utils/benches

COPY math_utils/src/lib.rs /workspace/math_utils/src/lib.rs
COPY math_utils/Cargo.toml /workspace/math_utils/Cargo.toml
COPY math_utils/benches/math_bench.rs /workspace/math_utils/benches/math_bench.rs
COPY math_utils/benches/prime_bench.rs /workspace/math_utils/benches/prime_bench.rs

WORKDIR /workspace/math_utils