FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install PyTorch and dependencies
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --break-system-packages pyyaml

# Create directories
RUN mkdir -p /workspace/training /workspace/config

WORKDIR /workspace/training

# Copy files
COPY train.py /workspace/training/train.py
COPY start_training.sh /workspace/training/start_training.sh
COPY requirements.txt /workspace/training/requirements.txt

RUN chmod +x /workspace/training/start_training.sh

# Copy config.yaml
COPY config.yaml /workspace/training/config.yaml

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
