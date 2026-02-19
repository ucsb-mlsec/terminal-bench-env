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
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dirmngr \
    gnupg2 \
    curl && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /etc/apt/keyrings/clickhouse-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" > /etc/apt/sources.list.d/clickhouse.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    clickhouse-server \
    clickhouse-client && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages clickhouse-connect

RUN mkdir -p /home/agent /var/lib/clickhouse /var/log/clickhouse-server /etc/clickhouse-server && \
    chown -R clickhouse:clickhouse /var/lib/clickhouse /var/log/clickhouse-server /etc/clickhouse-server

COPY init_clickhouse.sql /docker-entrypoint-initdb.d/init_clickhouse.sql

WORKDIR /home/agent