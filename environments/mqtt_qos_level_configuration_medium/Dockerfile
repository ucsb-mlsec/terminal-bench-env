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

RUN pip3 install --no-cache-dir --break-system-packages paho-mqtt

RUN mkdir -p /home/agent/sensor_network /home/agent/solution

COPY temperature_sensor.py /home/agent/sensor_network/temperature_sensor.py
COPY alert_monitor.py /home/agent/sensor_network/alert_monitor.py
COPY data_logger.py /home/agent/sensor_network/data_logger.py
COPY config/mqtt_config.json /home/agent/sensor_network/mqtt_config.json

WORKDIR /home/agent