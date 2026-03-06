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
    linux-tools-common \
    linux-tools-generic \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y linux-tools-generic linux-tools-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app /workspace/data

COPY app/data_processor /workspace/app/data_processor
COPY data/test_input.dat /workspace/app/test_input.dat

RUN chmod +x /workspace/app/data_processor

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
