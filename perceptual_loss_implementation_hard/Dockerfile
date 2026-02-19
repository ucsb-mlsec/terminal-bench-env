FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/vision_training/logs/experiment_1 \
    /workspace/vision_training/logs/experiment_2 \
    /workspace/vision_training/logs/experiment_3 \
    /workspace/vision_training/logs/experiment_4 \
    /workspace/vision_training/logs/experiment_5 \
    /workspace/vision_training/logs/experiment_6 \
    /workspace/vision_training/logs/experiment_7 \
    /workspace/vision_training/logs/experiment_8 \
    /workspace/vision_training/logs/experiment_9 \
    /workspace/vision_training/logs/experiment_10

COPY data/experiment_1/metrics.json /workspace/vision_training/logs/experiment_1/metrics.json
COPY data/experiment_1/config.json /workspace/vision_training/logs/experiment_1/config.json
COPY data/experiment_2/metrics.json /workspace/vision_training/logs/experiment_2/metrics.json
COPY data/experiment_2/config.json /workspace/vision_training/logs/experiment_2/config.json
COPY data/experiment_3/metrics.json /workspace/vision_training/logs/experiment_3/metrics.json
COPY data/experiment_3/config.json /workspace/vision_training/logs/experiment_3/config.json
COPY data/experiment_4/metrics.json /workspace/vision_training/logs/experiment_4/metrics.json
COPY data/experiment_4/config.json /workspace/vision_training/logs/experiment_4/config.json
COPY data/experiment_5/metrics.json /workspace/vision_training/logs/experiment_5/metrics.json
COPY data/experiment_5/config.json /workspace/vision_training/logs/experiment_5/config.json
COPY data/experiment_6/metrics.json /workspace/vision_training/logs/experiment_6/metrics.json
COPY data/experiment_6/config.json /workspace/vision_training/logs/experiment_6/config.json
COPY data/experiment_7/metrics.json /workspace/vision_training/logs/experiment_7/metrics.json
COPY data/experiment_7/config.json /workspace/vision_training/logs/experiment_7/config.json
COPY data/experiment_8/metrics.json /workspace/vision_training/logs/experiment_8/metrics.json
COPY data/experiment_8/config.json /workspace/vision_training/logs/experiment_8/config.json
COPY data/experiment_9/metrics.json /workspace/vision_training/logs/experiment_9/metrics.json
COPY data/experiment_9/config.json /workspace/vision_training/logs/experiment_9/config.json
COPY data/experiment_10/metrics.json /workspace/vision_training/logs/experiment_10/metrics.json
COPY data/experiment_10/config.json /workspace/vision_training/logs/experiment_10/config.json

WORKDIR /workspace