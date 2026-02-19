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
    protobuf-compiler \
    python3-protobuf && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages protobuf

RUN mkdir -p /data/sensor_telemetry /schemas /output

COPY schemas/sensor.proto /schemas/sensor.proto
COPY data/sensor_telemetry/sensor_001.pb /data/sensor_telemetry/sensor_001.pb
COPY data/sensor_telemetry/sensor_002.pb /data/sensor_telemetry/sensor_002.pb
COPY data/sensor_telemetry/sensor_003.pb /data/sensor_telemetry/sensor_003.pb
COPY data/sensor_telemetry/sensor_004.pb /data/sensor_telemetry/sensor_004.pb
COPY data/sensor_telemetry/sensor_005.pb /data/sensor_telemetry/sensor_005.pb

WORKDIR /workspace