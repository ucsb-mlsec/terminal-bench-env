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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages torch torchvision numpy

RUN mkdir -p /workspace/ml_project/data

COPY model.py /workspace/ml_project/model.py
COPY config/config.json /workspace/ml_project/config.json
COPY data/train_images.npy /workspace/ml_project/data/train_images.npy
COPY data/train_labels.npy /workspace/ml_project/data/train_labels.npy

WORKDIR /workspace/ml_project