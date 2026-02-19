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
    haproxy \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /config /data /home/agent

COPY config/haproxy.cfg /config/haproxy.cfg
COPY data/haproxy_traffic.log /data/haproxy_traffic.log
COPY data/backend_server.log /data/backend_server.log

WORKDIR /home/agent