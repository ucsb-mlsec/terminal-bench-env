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
    openjdk-11-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyspark

RUN mkdir -p /data/orders /app /test /solution

COPY data/orders/orders_batch1.csv /data/orders/orders_batch1.csv
COPY data/orders/orders_batch2.csv /data/orders/orders_batch2.csv
COPY analytics.py /app/analytics.py
COPY test/add_new_orders.sh /test/add_new_orders.sh

RUN chmod +x /test/add_new_orders.sh

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
