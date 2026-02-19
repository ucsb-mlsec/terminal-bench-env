FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/app_deps /tmp

COPY data/web_server.deps /opt/app_deps/web_server.deps
COPY data/database_client.deps /opt/app_deps/database_client.deps
COPY data/api_gateway.deps /opt/app_deps/api_gateway.deps
COPY data/message_queue.deps /opt/app_deps/message_queue.deps
COPY data/cache_service.deps /opt/app_deps/cache_service.deps

WORKDIR /opt/app_deps