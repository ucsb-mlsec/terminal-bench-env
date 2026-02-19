FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Agda and required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    agda \
    agda-stdlib \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/verify/inputs /workspace/data/verify/inputs

# Copy generated files to their expected locations
COPY verify/proof.agda /workspace/verify/proof.agda
COPY data/verify/inputs/input1.txt /workspace/verify/inputs/input1.txt
COPY data/verify/inputs/input2.txt /workspace/verify/inputs/input2.txt
COPY data/verify/inputs/input4.txt /workspace/verify/inputs/input4.txt

# Set working directory
WORKDIR /workspace