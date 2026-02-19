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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y rabbitmq-server && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pika

RUN mkdir -p /home/agent/alert_system

COPY alert_system/broker.py /home/agent/alert_system/broker.py
COPY alert_system/station_1.py /home/agent/alert_system/station_1.py
COPY alert_system/station_2.py /home/agent/alert_system/station_2.py
COPY alert_system/station_3.py /home/agent/alert_system/station_3.py
COPY alert_system/send_alert.py /home/agent/alert_system/send_alert.py

WORKDIR /home/agent/alert_system

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
