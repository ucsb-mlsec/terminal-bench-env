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
    apt-get install -y gdb && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc libc6-dbg && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/bin /workspace/src /workspace/data

COPY src/analyzer.c /workspace/src/analyzer.c
COPY bin/data_analyzer /workspace/bin/data_analyzer
COPY data/input.dat /workspace/data/input.dat

RUN chmod +x /workspace/bin/data_analyzer

WORKDIR /workspace