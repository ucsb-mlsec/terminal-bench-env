FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Agda and dependencies
RUN apt-get update && \
    apt-get install -y \
    agda \
    agda-stdlib \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y agda && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/agda-project

# Copy project files to their expected locations
COPY CollatzProof.agda /workspace/agda-project/CollatzProof.agda
COPY agda-lib /workspace/agda-project/agda-lib
COPY README.md /workspace/agda-project/README.md

# Set working directory
WORKDIR /workspace/agda-project