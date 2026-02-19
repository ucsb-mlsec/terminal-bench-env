FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/datalake/sales/year=2023/month=01 \
    /workspace/datalake/sales/year=2023/month=02 \
    /workspace/datalake/sales/year=2024/month=01 \
    /workspace/datalake/logs/date=2024-01-15 \
    /workspace/datalake/logs/date=2024-01-16 \
    /workspace/datalake/events/region=us-east/type=click \
    /workspace/datalake/events/region=us-east/type=view \
    /workspace/datalake/events/region=eu-west/type=click \
    /workspace/datalake/raw_data \
    /workspace/datalake/empty_dataset/year=2024 \
    /workspace/datalake/metrics/category=performance \
    /workspace/datalake/metrics/category=availability \
    /tmp

COPY data/datalake/sales/year=2023/month=01/data_001.parquet /workspace/datalake/sales/year=2023/month=01/data_001.parquet
COPY data/datalake/sales/year=2023/month=01/data_002.parquet /workspace/datalake/sales/year=2023/month=01/data_002.parquet
COPY data/datalake/sales/year=2023/month=02/data_001.parquet /workspace/datalake/sales/year=2023/month=02/data_001.parquet
COPY data/datalake/sales/year=2024/month=01/data_001.parquet /workspace/datalake/sales/year=2024/month=01/data_001.parquet
COPY data/datalake/logs/date=2024-01-15/log_001.json /workspace/datalake/logs/date=2024-01-15/log_001.json
COPY data/datalake/logs/date=2024-01-15/log_002.json /workspace/datalake/logs/date=2024-01-15/log_002.json
COPY data/datalake/logs/date=2024-01-16/log_001.json /workspace/datalake/logs/date=2024-01-16/log_001.json
COPY data/datalake/events/region=us-east/type=click/event_001.csv /workspace/datalake/events/region=us-east/type=click/event_001.csv
COPY data/datalake/events/region=us-east/type=click/event_002.csv /workspace/datalake/events/region=us-east/type=click/event_002.csv
COPY data/datalake/events/region=us-east/type=view/event_001.csv /workspace/datalake/events/region=us-east/type=view/event_001.csv
COPY data/datalake/events/region=eu-west/type=click/event_001.csv /workspace/datalake/events/region=eu-west/type=click/event_001.csv
COPY data/datalake/raw_data/file001.json /workspace/datalake/raw_data/file001.json
COPY data/datalake/raw_data/file002.json /workspace/datalake/raw_data/file002.json
COPY data/datalake/empty_dataset/year=2024/.gitkeep /workspace/datalake/empty_dataset/year=2024/.gitkeep
COPY data/datalake/metrics/category=performance/metric_001.parquet /workspace/datalake/metrics/category=performance/metric_001.parquet
COPY data/datalake/metrics/category=performance/metric_002.parquet /workspace/datalake/metrics/category=performance/metric_002.parquet
COPY data/datalake/metrics/category=availability/metric_001.parquet /workspace/datalake/metrics/category=availability/metric_001.parquet

WORKDIR /workspace