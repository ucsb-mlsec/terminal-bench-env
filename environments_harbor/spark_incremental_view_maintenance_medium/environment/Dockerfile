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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyarrow

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    pyarrow \
    fastparquet

RUN mkdir -p /data/transactions /data/materialized_view /app /solution

COPY data/transactions/2024-01-01.parquet /data/transactions/2024-01-01.parquet
COPY data/transactions/2024-01-02.parquet /data/transactions/2024-01-02.parquet
COPY data/transactions/2024-01-03.parquet /data/transactions/2024-01-03.parquet
COPY data/transactions/2024-01-04.parquet /data/transactions/2024-01-04.parquet
COPY data/transactions/2024-01-05.parquet /data/transactions/2024-01-05.parquet
COPY data/materialized_view/view.parquet /data/materialized_view/view.parquet
COPY rebuild_view.py /app/rebuild_view.py

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
