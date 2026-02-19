FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/mysql /opt/database/schema /tmp

COPY data/mysql-slow.log /var/log/mysql/mysql-slow.log
COPY data/schema/users.sql /opt/database/schema/users.sql
COPY data/schema/orders.sql /opt/database/schema/orders.sql
COPY data/schema/products.sql /opt/database/schema/products.sql
COPY data/schema/order_items.sql /opt/database/schema/order_items.sql
COPY data/schema/customers.sql /opt/database/schema/customers.sql
COPY data/schema/transactions.sql /opt/database/schema/transactions.sql

WORKDIR /workspace