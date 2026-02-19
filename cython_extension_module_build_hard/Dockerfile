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
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential && \
    pip3 install --no-cache-dir --break-system-packages numpy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages cython

RUN mkdir -p /workspace

COPY compute_core.pyx /workspace/compute_core.pyx
COPY test_runner.py /workspace/test_runner.py
COPY requirements.txt /workspace/requirements.txt
COPY data/data.csv /workspace/data.csv

WORKDIR /workspace

RUN if [ -s requirements.txt ]; then pip3 install --no-cache-dir --break-system-packages -r requirements.txt; fi