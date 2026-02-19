FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/ml_checkpoints

COPY data/ml_checkpoints/checkpoint_epoch_25.json /workspace/ml_checkpoints/checkpoint_epoch_25.json
COPY data/ml_checkpoints/checkpoint_epoch_35.json /workspace/ml_checkpoints/checkpoint_epoch_35.json
COPY data/ml_checkpoints/checkpoint_epoch_50.json /workspace/ml_checkpoints/checkpoint_epoch_50.json
COPY data/ml_checkpoints/checkpoint_epoch_60.json /workspace/ml_checkpoints/checkpoint_epoch_60.json
COPY data/ml_checkpoints/checkpoint_epoch_75.json /workspace/ml_checkpoints/checkpoint_epoch_75.json
COPY data/ml_checkpoints/checkpoint_epoch_80.json /workspace/ml_checkpoints/checkpoint_epoch_80.json
COPY data/ml_checkpoints/checkpoint_epoch_90.json /workspace/ml_checkpoints/checkpoint_epoch_90.json
COPY data/ml_checkpoints/checkpoint_epoch_95.json /workspace/ml_checkpoints/checkpoint_epoch_95.json

WORKDIR /workspace