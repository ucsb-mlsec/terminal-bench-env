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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages torch numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    torch \
    pandas \
    numpy

RUN mkdir -p /workspace/sensor_data /workspace/solution

COPY data/sensor_data/device_001.csv /workspace/sensor_data/device_001.csv
COPY data/sensor_data/device_002.csv /workspace/sensor_data/device_002.csv
COPY data/sensor_data/device_003.csv /workspace/sensor_data/device_003.csv
COPY data/sensor_data/device_004.csv /workspace/sensor_data/device_004.csv

WORKDIR /workspace