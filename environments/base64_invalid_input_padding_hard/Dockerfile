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
    gzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/logs/distributed_auth/monitoring && \
    mkdir -p /var/logs/distributed_auth/security && \
    mkdir -p /home/user

COPY data/var/logs/distributed_auth/auth_service.log /var/logs/distributed_auth/auth_service.log
COPY data/var/logs/distributed_auth/api_gateway.log /var/logs/distributed_auth/api_gateway.log
COPY data/var/logs/distributed_auth/database_service.log /var/logs/distributed_auth/database_service.log
COPY data/var/logs/distributed_auth/cache_service.log.gz /var/logs/distributed_auth/cache_service.log.gz
COPY data/var/logs/distributed_auth/monitoring/monitor.log /var/logs/distributed_auth/monitoring/monitor.log
COPY data/var/logs/distributed_auth/security/security.log /var/logs/distributed_auth/security/security.log
COPY data/var/logs/distributed_auth/user_service.log /var/logs/distributed_auth/user_service.log

WORKDIR /home/user