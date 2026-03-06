FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/configs /solution

COPY data/configs/param_set_1.txt /data/configs/param_set_1.txt
COPY data/configs/param_set_2.txt /data/configs/param_set_2.txt
COPY data/configs/param_set_3.txt /data/configs/param_set_3.txt
COPY data/configs/validation_rules.txt /data/configs/validation_rules.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
