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
    apt-get install -y \
    tshark \
    openssl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y tshark openssl && \
    pip3 install --no-cache-dir --break-system-packages scapy && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/server \
    /opt/client \
    /var/log/ssl_failures \
    /solution

COPY config/nginx.conf /opt/server/nginx.conf
COPY data/ssl_handshake_failure.pcap.txt /var/log/ssl_failures/ssl_handshake_failure.pcap.txt
COPY data/client_info.txt /opt/client/client_info.txt

WORKDIR /workspace