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
    openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyspark pandas

RUN mkdir -p /workspace /workspace/data /workspace/config

COPY data/customers.csv /workspace/data/customers.csv
COPY data/transactions.csv /workspace/data/transactions.csv
COPY data/partition_stats.json /workspace/data/partition_stats.json
COPY config/spark_app.conf /workspace/config/spark_app.conf
COPY join_query.sql /workspace/join_query.sql

WORKDIR /workspace