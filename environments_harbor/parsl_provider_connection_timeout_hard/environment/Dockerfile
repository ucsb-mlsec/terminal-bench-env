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

RUN pip3 install --no-cache-dir --break-system-packages parsl pandas

RUN mkdir -p /workspace/data /solution

COPY data/sensor_001.csv /workspace/data/sensor_001.csv
COPY data/sensor_002.csv /workspace/data/sensor_002.csv
COPY data/sensor_003.csv /workspace/data/sensor_003.csv
COPY data/sensor_004.csv /workspace/data/sensor_004.csv
COPY data/sensor_005.csv /workspace/data/sensor_005.csv
COPY pipeline.py /workspace/pipeline.py
COPY verify.py /workspace/verify.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
