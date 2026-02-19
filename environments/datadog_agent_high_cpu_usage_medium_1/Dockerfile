FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/datadog-agent/conf.d /root

COPY config/datadog-agent/conf.d/disk.yaml /etc/datadog-agent/conf.d/disk.yaml
COPY config/datadog-agent/conf.d/cpu.yaml /etc/datadog-agent/conf.d/cpu.yaml
COPY config/datadog-agent/conf.d/process.yaml /etc/datadog-agent/conf.d/process.yaml
COPY config/datadog-agent/conf.d/network.yaml /etc/datadog-agent/conf.d/network.yaml
COPY config/datadog-agent/conf.d/nginx.yaml /etc/datadog-agent/conf.d/nginx.yaml
COPY config/datadog-agent/datadog.yaml /etc/datadog-agent/datadog.yaml

WORKDIR /root