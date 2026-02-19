FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install SPIN model checker and build tools
RUN apt-get update && \
    apt-get install -y \
    spin \
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y spin gcc make && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/protocol_model

# Copy generated files to their expected locations
COPY data/protocol_spec.txt /workspace/protocol_model/protocol_spec.txt
COPY Makefile /workspace/protocol_model/Makefile
COPY token_ring.pml /workspace/protocol_model/token_ring.pml

# Set working directory to match task requirements
WORKDIR /workspace/protocol_model