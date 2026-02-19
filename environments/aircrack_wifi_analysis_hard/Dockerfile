FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/handshake_logs

COPY data/handshake_logs/network_01.log /workspace/handshake_logs/network_01.log
COPY data/handshake_logs/network_02.log /workspace/handshake_logs/network_02.log
COPY data/handshake_logs/network_03.log /workspace/handshake_logs/network_03.log
COPY data/handshake_logs/network_04.log /workspace/handshake_logs/network_04.log
COPY data/handshake_logs/network_05.log /workspace/handshake_logs/network_05.log
COPY data/handshake_logs/network_06.log /workspace/handshake_logs/network_06.log
COPY data/common_passwords.txt /workspace/common_passwords.txt

WORKDIR /workspace