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
    apt-get install -y tcpdump && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/network_monitor/configs /home/agent

COPY config/web_traffic.conf /opt/network_monitor/configs/web_traffic.conf
COPY config/dns_monitor.conf /opt/network_monitor/configs/dns_monitor.conf
COPY config/database_traffic.conf /opt/network_monitor/configs/database_traffic.conf

WORKDIR /home/agent