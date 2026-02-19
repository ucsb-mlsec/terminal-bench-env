FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/rabbitmq_incident /tmp

COPY config/rabbitmq.conf /tmp/rabbitmq_incident/rabbitmq.conf
COPY data/queue_stats.log /tmp/rabbitmq_incident/queue_stats.log
COPY data/memory_report.txt /tmp/rabbitmq_incident/memory_report.txt

WORKDIR /tmp