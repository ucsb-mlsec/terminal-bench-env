FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/monitoring/conf.d /opt/monitoring/data /root

COPY config/agent.yaml /opt/monitoring/agent.yaml
COPY config/conf.d/cpu_check.yaml /opt/monitoring/conf.d/cpu_check.yaml
COPY config/conf.d/memory_check.yaml /opt/monitoring/conf.d/memory_check.yaml
COPY config/conf.d/disk_check.yaml /opt/monitoring/conf.d/disk_check.yaml
COPY config/conf.d/network_check.yaml /opt/monitoring/conf.d/network_check.yaml
COPY data/process_snapshot.log /opt/monitoring/process_snapshot.log
COPY data/agent.log /opt/monitoring/agent.log

WORKDIR /root