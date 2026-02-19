FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/db/production \
    && mkdir -p /var/backups/remote \
    && mkdir -p /var/log \
    && mkdir -p /usr/local/bin

COPY data/production/customers.db /var/db/production/customers.db
COPY data/production/orders.db /var/db/production/orders.db
COPY data/production/inventory.db /var/db/production/inventory.db
COPY data/production/analytics.db /var/db/production/analytics.db
COPY data/production/users.db /var/db/production/users.db

WORKDIR /var/db/production