FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /solution \
    && mkdir -p /config \
    && mkdir -p /etc/profile.d \
    && mkdir -p /etc/systemd/system \
    && mkdir -p /home/user \
    && mkdir -p /usr/local/cuda

# Copy all generated files to their appropriate locations
COPY config/mps-environment.conf /config/mps-environment.conf
COPY etc/profile.d/cuda-mps.sh /etc/profile.d/cuda-mps.sh
COPY etc/systemd/system/nvidia-mps.service /etc/systemd/system/nvidia-mps.service
COPY home/user/.bashrc /home/user/.bashrc
COPY usr/local/cuda/README.txt /usr/local/cuda/README.txt

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
