FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pyjwt

RUN mkdir -p /data/requests /config /output

COPY data/requests/requests_001.json /data/requests/requests_001.json
COPY data/requests/requests_002.json /data/requests/requests_002.json
COPY data/requests/requests_003.json /data/requests/requests_003.json
COPY config/secret.key /config/secret.key

WORKDIR /workspace