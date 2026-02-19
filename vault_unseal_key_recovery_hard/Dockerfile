FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install system utilities
RUN apt-get update && \
    apt-get install -y \
    vim \
    binutils \
    bsdmainutils \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /data /config

# Copy data files
COPY data/memory_dump.bin /data/memory_dump.bin
COPY data/vault.log /data/vault.log
COPY data/system.log /data/system.log
COPY data/vault_audit.log /data/vault_audit.log
COPY data/process_dump.txt /data/process_dump.txt
COPY data/temp_keys.txt /data/temp_keys.txt
COPY data/core_dump_fragment.hex /data/core_dump_fragment.hex

# Copy config files
COPY config/vault.hcl /config/vault.hcl

WORKDIR /workspace