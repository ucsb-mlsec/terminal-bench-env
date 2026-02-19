FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/network_logs /tmp

COPY data/server1_diagnostics.log /tmp/network_logs/server1_diagnostics.log
COPY data/server2_diagnostics.log /tmp/network_logs/server2_diagnostics.log
COPY data/server3_diagnostics.log /tmp/network_logs/server3_diagnostics.log
COPY data/server4_diagnostics.log /tmp/network_logs/server4_diagnostics.log

WORKDIR /workspace