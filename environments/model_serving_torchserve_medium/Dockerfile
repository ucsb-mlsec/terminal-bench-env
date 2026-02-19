FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /config /tmp

COPY config/config.properties /config/config.properties
COPY config/model-store.properties /config/model-store.properties
COPY config/logging.properties /config/logging.properties
COPY config/runtime.properties /config/runtime.properties
COPY config/server.properties /config/server.properties

WORKDIR /workspace