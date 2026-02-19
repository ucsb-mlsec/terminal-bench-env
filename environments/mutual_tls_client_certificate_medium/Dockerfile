FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/mtls_server/certs

COPY server.py /tmp/mtls_server/server.py
COPY certs/ca.crt /tmp/mtls_server/certs/ca.crt
COPY certs/server.crt /tmp/mtls_server/certs/server.crt
COPY certs/server.key /tmp/mtls_server/certs/server.key
COPY certs/client.crt /tmp/mtls_server/certs/client.crt
COPY certs/client.key /tmp/mtls_server/certs/client.key

WORKDIR /tmp/mtls_server