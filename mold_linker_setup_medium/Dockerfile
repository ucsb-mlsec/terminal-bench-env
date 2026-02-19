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
    build-essential \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/test_project /usr/local/bin

RUN wget -q https://github.com/rui314/mold/releases/download/v2.4.0/mold-2.4.0-x86_64-linux.tar.gz -O /tmp/mold.tar.gz && \
    tar -xzf /tmp/mold.tar.gz -C /tmp && \
    cp /tmp/mold-2.4.0-x86_64-linux/bin/mold /usr/local/bin/mold && \
    chmod +x /usr/local/bin/mold && \
    rm -rf /tmp/mold.tar.gz /tmp/mold-2.4.0-x86_64-linux

COPY test_project/main.cpp /workspace/test_project/main.cpp
COPY test_project/Makefile /workspace/test_project/Makefile

WORKDIR /workspace/test_project