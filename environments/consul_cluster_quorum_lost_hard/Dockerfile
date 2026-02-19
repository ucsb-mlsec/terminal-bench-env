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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/consul.d /var/lib/consul /root

COPY config/server-1.json /etc/consul.d/server-1.json
COPY config/server-2.json /etc/consul.d/server-2.json
COPY config/server-3.json /etc/consul.d/server-3.json
COPY config/server-4.json /etc/consul.d/server-4.json
COPY config/server-5.json /etc/consul.d/server-5.json

WORKDIR /root