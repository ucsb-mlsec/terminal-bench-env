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
    apt-get install -y \
    build-essential \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/sensor_system/input

COPY sensor_data.h /workspace/sensor_system/sensor_data.h
COPY aggregator.c /workspace/sensor_system/aggregator.c
COPY Makefile /workspace/sensor_system/Makefile
COPY data/sensors.txt /workspace/sensor_system/input/sensors.txt

WORKDIR /workspace/sensor_system

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
