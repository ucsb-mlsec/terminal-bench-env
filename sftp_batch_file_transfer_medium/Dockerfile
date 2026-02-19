FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/incoming_files

COPY data/incoming_files/partner_001_20240115.csv /home/agent/incoming_files/partner_001_20240115.csv
COPY data/incoming_files/partner_002_20240115.csv /home/agent/incoming_files/partner_002_20240115.csv
COPY data/incoming_files/partner_003_20240115.csv /home/agent/incoming_files/partner_003_20240115.csv
COPY data/incoming_files/partner_004_20240115.csv /home/agent/incoming_files/partner_004_20240115.csv
COPY data/incoming_files/partner_006_20240115.csv /home/agent/incoming_files/partner_006_20240115.csv
COPY data/incoming_files/partner_007_20240115.csv /home/agent/incoming_files/partner_007_20240115.csv
COPY data/incoming_files/partner_008_20240115.csv /home/agent/incoming_files/partner_008_20240115.csv
COPY data/incoming_files/partner_009_20240115.csv /home/agent/incoming_files/partner_009_20240115.csv
COPY data/incoming_files/partner_010_20240115.csv /home/agent/incoming_files/partner_010_20240115.csv

WORKDIR /home/agent