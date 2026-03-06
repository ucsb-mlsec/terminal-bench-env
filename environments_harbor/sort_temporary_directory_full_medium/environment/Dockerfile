FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y coreutils && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/data_pipeline /data/input /data/output /data/workspace

COPY data/transactions.csv /data/input/transactions.csv
COPY process_transactions.sh /opt/data_pipeline/process_transactions.sh

RUN chmod +x /opt/data_pipeline/process_transactions.sh

WORKDIR /data

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
