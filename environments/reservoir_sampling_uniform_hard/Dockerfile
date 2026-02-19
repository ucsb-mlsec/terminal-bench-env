FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/sensor_streams/test /home/agent

COPY data/sensor_streams/temperature_sensors.txt /tmp/sensor_streams/temperature_sensors.txt
COPY data/sensor_streams/pressure_sensors.txt /tmp/sensor_streams/pressure_sensors.txt
COPY data/sensor_streams/humidity_sensors.txt /tmp/sensor_streams/humidity_sensors.txt
COPY data/sensor_streams/test/small_stream.txt /tmp/sensor_streams/test/small_stream.txt
COPY data/sensor_streams/test/medium_stream.txt /tmp/sensor_streams/test/medium_stream.txt
COPY data/sensor_streams/test/large_stream.txt /tmp/sensor_streams/test/large_stream.txt

WORKDIR /home/agent