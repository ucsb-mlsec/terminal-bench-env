FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/datadog-agent/conf.d/apache.d && \
    mkdir -p /etc/datadog-agent/conf.d/postgres.d && \
    mkdir -p /etc/datadog-agent/conf.d/mysql.d && \
    mkdir -p /etc/datadog-agent/conf.d/nginx.d && \
    mkdir -p /etc/datadog-agent/conf.d/disk.d && \
    mkdir -p /var/log/datadog && \
    mkdir -p /tmp

COPY config/datadog-agent/datadog.yaml /etc/datadog-agent/datadog.yaml
COPY config/datadog-agent/conf.d/apache.d/conf.yaml /etc/datadog-agent/conf.d/apache.d/conf.yaml
COPY config/datadog-agent/conf.d/postgres.d/conf.yaml /etc/datadog-agent/conf.d/postgres.d/conf.yaml
COPY config/datadog-agent/conf.d/mysql.d/conf.yaml /etc/datadog-agent/conf.d/mysql.d/conf.yaml
COPY config/datadog-agent/conf.d/nginx.d/conf.yaml /etc/datadog-agent/conf.d/nginx.d/conf.yaml
COPY config/datadog-agent/conf.d/disk.d/conf.yaml /etc/datadog-agent/conf.d/disk.d/conf.yaml
COPY data/datadog/agent.log /var/log/datadog/agent.log
COPY data/datadog/collector.log /var/log/datadog/collector.log

WORKDIR /tmp