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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages python-dateutil

RUN mkdir -p /workspace/events/iot_sensors && \
    mkdir -p /workspace/events/app_logs && \
    mkdir -p /workspace/events/webhooks && \
    mkdir -p /workspace/output

COPY data/events/iot_sensors/sensor_001.json /workspace/events/iot_sensors/sensor_001.json
COPY data/events/iot_sensors/sensor_002.json /workspace/events/iot_sensors/sensor_002.json
COPY data/events/iot_sensors/sensor_003.json /workspace/events/iot_sensors/sensor_003.json
COPY data/events/iot_sensors/sensor_004.json /workspace/events/iot_sensors/sensor_004.json
COPY data/events/iot_sensors/sensor_005.json /workspace/events/iot_sensors/sensor_005.json
COPY data/events/iot_sensors/sensor_006.json /workspace/events/iot_sensors/sensor_006.json

COPY data/events/app_logs/app_001.json /workspace/events/app_logs/app_001.json
COPY data/events/app_logs/app_002.json /workspace/events/app_logs/app_002.json
COPY data/events/app_logs/app_003.json /workspace/events/app_logs/app_003.json
COPY data/events/app_logs/app_004.json /workspace/events/app_logs/app_004.json
COPY data/events/app_logs/app_005.json /workspace/events/app_logs/app_005.json

COPY data/events/webhooks/webhook_001.json /workspace/events/webhooks/webhook_001.json
COPY data/events/webhooks/webhook_002.json /workspace/events/webhooks/webhook_002.json
COPY data/events/webhooks/webhook_003.json /workspace/events/webhooks/webhook_003.json
COPY data/events/webhooks/webhook_004.json /workspace/events/webhooks/webhook_004.json
COPY data/events/webhooks/webhook_005.json /workspace/events/webhooks/webhook_005.json
COPY data/events/webhooks/webhook_006.json /workspace/events/webhooks/webhook_006.json

WORKDIR /workspace