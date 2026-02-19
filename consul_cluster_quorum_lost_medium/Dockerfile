FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/consul /solution

COPY config/server1.json /etc/consul/server1.json
COPY config/server2.json /etc/consul/server2.json
COPY config/server3.json /etc/consul/server3.json
COPY config/server4.json /etc/consul/server4.json
COPY config/server5.json /etc/consul/server5.json
COPY config/peers.json /etc/consul/peers.json

WORKDIR /workspace