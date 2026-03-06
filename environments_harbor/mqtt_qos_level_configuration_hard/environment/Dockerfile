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
    apt-get install -y mosquitto mosquitto-clients && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml paho-mqtt

RUN mkdir -p /config /tmp

COPY config/critical_alert_publisher.conf /config/critical_alert_publisher.conf
COPY config/sensor_reading_publisher.conf /config/sensor_reading_publisher.conf
COPY config/heartbeat_publisher.conf /config/heartbeat_publisher.conf
COPY config/critical_alert_subscriber.conf /config/critical_alert_subscriber.conf
COPY config/dashboard_subscriber.conf /config/dashboard_subscriber.conf
COPY config/health_check_subscriber.conf /config/health_check_subscriber.conf
COPY config/mosquitto.conf /config/mosquitto.conf
COPY config/sensor_network.yaml /config/sensor_network.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
