FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy

RUN mkdir -p /data/experiments /solution/output

COPY data/sensor_readings_2024_01.npy /data/experiments/sensor_readings_2024_01.npy
COPY data/sensor_readings_2024_02.npy /data/experiments/sensor_readings_2024_02.npy
COPY data/sensor_readings_2024_03.npy /data/experiments/sensor_readings_2024_03.npy

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
