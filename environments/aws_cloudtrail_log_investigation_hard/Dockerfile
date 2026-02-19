FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cloudtrail_logs

COPY data/cloudtrail_logs/cloudtrail-2024-01-20-18.json.gz /workspace/cloudtrail_logs/
COPY data/cloudtrail_logs/cloudtrail-2024-01-20-19.json.gz /workspace/cloudtrail_logs/
COPY data/cloudtrail_logs/cloudtrail-2024-01-20-20.json.gz /workspace/cloudtrail_logs/
COPY data/cloudtrail_logs/cloudtrail-2024-01-20-21.json.gz /workspace/cloudtrail_logs/
COPY data/cloudtrail_logs/cloudtrail-2024-01-20-22.json.gz /workspace/cloudtrail_logs/
COPY data/cloudtrail_logs/cloudtrail-2024-01-20-23.json.gz /workspace/cloudtrail_logs/

WORKDIR /workspace