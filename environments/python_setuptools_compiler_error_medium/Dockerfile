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
    python3-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc python3-dev && \
    pip3 install --no-cache-dir --break-system-packages setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathops /tmp

COPY mathops/setup.py /workspace/mathops/setup.py
COPY mathops/mathops_ext.c /workspace/mathops/mathops_ext.c
COPY mathops/README.md /workspace/mathops/README.md

WORKDIR /workspace