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

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pillow \
    pyyaml

RUN mkdir -p /workspace/config /workspace/validation_data /workspace/solution /workspace/data

COPY config/noise_schedule.yaml /workspace/config/noise_schedule.yaml
COPY data/validation_data/image_001.png /workspace/validation_data/image_001.png
COPY data/validation_data/image_002.png /workspace/validation_data/image_002.png
COPY data/validation_data/image_003.png /workspace/validation_data/image_003.png
COPY data/validation_data/image_004.png /workspace/validation_data/image_004.png
COPY data/validation_data/image_005.png /workspace/validation_data/image_005.png
COPY data/training_logs.txt /workspace/data/training_logs.txt
COPY data/model_checkpoint.pth /workspace/data/model_checkpoint.pth

WORKDIR /workspace