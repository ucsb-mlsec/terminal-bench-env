FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data /workspace/output /workspace/output/iceberg_table_evolution_medium/files/data

RUN pip3 install --no-cache-dir --break-system-packages pandas pyarrow

COPY data/inventory_snapshot.parquet /workspace/data/inventory_snapshot.parquet
COPY data/inventory_snapshot.parquet /workspace/output/iceberg_table_evolution_medium/files/data/inventory_snapshot.parquet

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
