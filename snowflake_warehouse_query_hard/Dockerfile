FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/warehouse_data

COPY data/warehouse_data/transactions.csv /home/user/warehouse_data/transactions.csv
COPY data/warehouse_data/customers.csv /home/user/warehouse_data/customers.csv
COPY data/warehouse_data/products.csv /home/user/warehouse_data/products.csv

RUN pip3 install --no-cache-dir --break-system-packages pandas

WORKDIR /home/user