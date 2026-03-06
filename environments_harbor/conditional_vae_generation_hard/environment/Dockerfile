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
RUN pip3 install --no-cache-dir --break-system-packages torch numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --break-system-packages \
    numpy matplotlib

RUN mkdir -p /data/shapes /tmp

COPY data/shapes/train_images.npy /data/shapes/train_images.npy
COPY data/shapes/train_labels.npy /data/shapes/train_labels.npy
COPY train_cvae.py /workspace/train_cvae.py
COPY test_cvae.py /workspace/test_cvae.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
