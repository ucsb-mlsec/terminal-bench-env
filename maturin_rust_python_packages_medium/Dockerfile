FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip curl build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Rust build dependencies
RUN pip3 install --no-cache-dir --break-system-packages setuptools-rust maturin

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Create workspace directory
RUN mkdir -p /workspace/mathops/src

WORKDIR /workspace

# Copy all files
COPY . /workspace/