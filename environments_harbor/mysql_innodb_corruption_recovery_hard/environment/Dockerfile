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
    apt-get install -y mysql-server && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-mysql-client default-mysql-server && \
    pip3 install --no-cache-dir --break-system-packages pymysql && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/backups/mysql_dumps /root

COPY data/customers.sql /var/backups/mysql_dumps/customers.sql
COPY data/orders.sql /var/backups/mysql_dumps/orders.sql
COPY data/products.sql /var/backups/mysql_dumps/products.sql

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
