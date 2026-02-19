FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /incident_data/consumer_logs /incident_data/configs /home

COPY data/incident_data/rabbitmq_queues.json /incident_data/rabbitmq_queues.json
COPY data/incident_data/rabbitmq_connections.json /incident_data/rabbitmq_connections.json
COPY data/incident_data/consumer_logs/payment_service.log /incident_data/consumer_logs/payment_service.log
COPY data/incident_data/consumer_logs/order_service.log /incident_data/consumer_logs/order_service.log
COPY data/incident_data/consumer_logs/email_service.log /incident_data/consumer_logs/email_service.log
COPY data/incident_data/configs/payment_service_config.yaml /incident_data/configs/payment_service_config.yaml
COPY data/incident_data/configs/order_service_config.yaml /incident_data/configs/order_service_config.yaml
COPY data/incident_data/configs/email_service_config.yaml /incident_data/configs/email_service_config.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /home