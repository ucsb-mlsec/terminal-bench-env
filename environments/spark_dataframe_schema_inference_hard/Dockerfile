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
    openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyspark

RUN mkdir -p /data/transactions /solution /output /workspace

COPY data/transactions/region_north.csv /data/transactions/region_north.csv
COPY data/transactions/region_south.csv /data/transactions/region_south.csv
COPY data/transactions/region_east.csv /data/transactions/region_east.csv
COPY data/transactions/region_west.csv /data/transactions/region_west.csv
COPY data/transactions/region_central.csv /data/transactions/region_central.csv

WORKDIR /workspace