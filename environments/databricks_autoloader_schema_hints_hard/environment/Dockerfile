FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/sensors/temperature /data/sensors/pressure /data/sensors/humidity /solution

COPY data/sensors/temperature/sensor_temp_001.json /data/sensors/temperature/sensor_temp_001.json
COPY data/sensors/temperature/sensor_temp_002.json /data/sensors/temperature/sensor_temp_002.json
COPY data/sensors/pressure/sensor_pressure_001.json /data/sensors/pressure/sensor_pressure_001.json
COPY data/sensors/pressure/sensor_pressure_002.json /data/sensors/pressure/sensor_pressure_002.json
COPY data/sensors/humidity/sensor_humidity_001.json /data/sensors/humidity/sensor_humidity_001.json
COPY data/sensors/humidity/sensor_humidity_002.json /data/sensors/humidity/sensor_humidity_002.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
