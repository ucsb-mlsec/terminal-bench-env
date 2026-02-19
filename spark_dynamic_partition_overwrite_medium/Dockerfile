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
    apt-get install -y openjdk-11-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyspark

RUN mkdir -p /workspace /data/sales /data/incoming /solution

COPY sales_pipeline.py /workspace/sales_pipeline.py
COPY data/incoming/daily_sales.csv /data/incoming/daily_sales.csv
COPY data/sales/region=US/date=2024-01-10/part-00000.csv /data/sales/region=US/date=2024-01-10/part-00000.csv
COPY data/sales/region=US/date=2024-01-12/part-00000.csv /data/sales/region=US/date=2024-01-12/part-00000.csv
COPY data/sales/region=EU/date=2024-01-10/part-00000.csv /data/sales/region=EU/date=2024-01-10/part-00000.csv
COPY data/sales/region=EU/date=2024-01-12/part-00000.csv /data/sales/region=EU/date=2024-01-12/part-00000.csv
COPY data/sales/region=EU/date=2024-01-15/part-00000.csv /data/sales/region=EU/date=2024-01-15/part-00000.csv
COPY data/sales/region=ASIA/date=2024-01-10/part-00000.csv /data/sales/region=ASIA/date=2024-01-10/part-00000.csv
COPY data/sales/region=ASIA/date=2024-01-12/part-00000.csv /data/sales/region=ASIA/date=2024-01-12/part-00000.csv
COPY data/sales/region=ASIA/date=2024-01-15/part-00000.csv /data/sales/region=ASIA/date=2024-01-15/part-00000.csv

WORKDIR /workspace