FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /vault /services /solution /data

COPY data/credentials.json /vault/credentials.json
COPY data/service-1.state /services/service-1.state
COPY data/service-2.state /services/service-2.state
COPY data/service-3.state /services/service-3.state
COPY data/rotation.log /vault/rotation.log

WORKDIR /solution