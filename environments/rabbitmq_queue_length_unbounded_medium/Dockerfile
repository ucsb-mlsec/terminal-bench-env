FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pika

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/rabbitmq_analysis/config /opt/rabbitmq_analysis/data /tmp

COPY config/rabbitmq.conf /opt/rabbitmq_analysis/rabbitmq.conf
COPY data/queue_stats.json /opt/rabbitmq_analysis/queue_stats.json
COPY config/producer_config.yaml /opt/rabbitmq_analysis/producer_config.yaml
COPY config/consumer_config.yaml /opt/rabbitmq_analysis/consumer_config.yaml
COPY consumer.py /opt/rabbitmq_analysis/consumer.py
COPY data/application.log /opt/rabbitmq_analysis/application.log

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /opt/rabbitmq_analysis