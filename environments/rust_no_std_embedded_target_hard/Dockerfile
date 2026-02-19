FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install all necessary packages in one layer
RUN apt-get update && \
    apt-get install -y python3-pip curl build-essential rustc cargo gcc binutils && \
    rm -rf /var/lib/apt/lists/*

# Create python symlink if it doesn't exist
RUN [ ! -e /usr/bin/python ] && ln -s /usr/bin/python3 /usr/bin/python || true

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install thumbv7em-none-eabi target for ARM Cortex-M4
RUN /root/.cargo/bin/rustup target add thumbv7em-none-eabi

# Create project directories
RUN mkdir -p /workspace/embedded-lib/.cargo /workspace/embedded-lib/src

# Copy project files
COPY Cargo.toml /workspace/embedded-lib/Cargo.toml
COPY .cargo/config.toml /workspace/embedded-lib/.cargo/config.toml
COPY cortex-m4.json /workspace/embedded-lib/cortex-m4.json
COPY src/lib.rs /workspace/embedded-lib/src/lib.rs

# Set working directory
WORKDIR /workspace/embedded-lib