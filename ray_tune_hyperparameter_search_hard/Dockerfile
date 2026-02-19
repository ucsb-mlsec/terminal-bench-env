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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages ray[tune]>=2.0

RUN pip3 install --no-cache-dir --break-system-packages \
    ray[tune] \
    numpy \
    scikit-learn \
    torch

RUN mkdir -p /workspace/ray_custom/data

COPY training_function.py /workspace/ray_custom/training_function.py
COPY data/historical_results.json /workspace/ray_custom/historical_results.json

WORKDIR /workspace/ray_custom