FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/locust_stats

COPY data/worker_1.csv /tmp/locust_stats/worker_1.csv
COPY data/worker_2.csv /tmp/locust_stats/worker_2.csv
COPY data/worker_3.csv /tmp/locust_stats/worker_3.csv
COPY data/worker_4.csv /tmp/locust_stats/worker_4.csv
COPY data/worker_5.csv /tmp/locust_stats/worker_5.csv

RUN pip3 install --no-cache-dir --break-system-packages pandas

WORKDIR /workspace