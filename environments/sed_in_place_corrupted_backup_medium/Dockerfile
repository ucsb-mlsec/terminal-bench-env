FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/configs

COPY configs/app.conf /workspace/configs/app.conf
COPY configs/app.conf.bak /workspace/configs/app.conf.bak
COPY configs/database.conf /workspace/configs/database.conf
COPY configs/database.conf.bak /workspace/configs/database.conf.bak
COPY configs/logging.conf /workspace/configs/logging.conf
COPY configs/network.conf /workspace/configs/network.conf
COPY configs/network.conf.bak /workspace/configs/network.conf.bak
COPY configs/redis.conf /workspace/configs/redis.conf
COPY configs/security.conf /workspace/configs/security.conf
COPY configs/security.conf.bak /workspace/configs/security.conf.bak
COPY configs/server.conf /workspace/configs/server.conf
COPY configs/storage.conf /workspace/configs/storage.conf

WORKDIR /workspace