FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install DuckDB and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/duckdb_cli-linux-amd64.zip -O /tmp/duckdb.zip && \
    unzip /tmp/duckdb.zip -d /tmp && \
    mv /tmp/duckdb /usr/bin/duckdb && \
    chmod +x /usr/bin/duckdb && \
    rm /tmp/duckdb.zip && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages for working with Parquet files
RUN pip3 install --no-cache-dir --break-system-packages pyarrow pandas duckdb

# Create required directories
RUN mkdir -p /data/orders/year=2022 \
    /data/orders/year=2023 \
    /solution/data \
    /solution

# Copy generated files to appropriate locations
COPY data/orders/year=2022/orders.parquet /data/orders/year=2022/orders.parquet
COPY data/orders/year=2023/orders.parquet /data/orders/year=2023/orders.parquet
COPY data/run_baseline.sh /data/run_baseline.sh
COPY data/expected_results.json /data/expected_results.json

# Make baseline script executable
RUN chmod +x /data/run_baseline.sh

WORKDIR /solution