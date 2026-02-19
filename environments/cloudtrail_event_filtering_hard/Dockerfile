FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cloudtrail_logs/production /workspace/cloudtrail_logs/development

COPY data/cloudtrail_logs/production/cloudtrail_prod_001.json /workspace/cloudtrail_logs/production/cloudtrail_prod_001.json
COPY data/cloudtrail_logs/production/cloudtrail_prod_003.json /workspace/cloudtrail_logs/production/cloudtrail_prod_003.json
COPY data/cloudtrail_logs/production/cloudtrail_prod_005.json /workspace/cloudtrail_logs/production/cloudtrail_prod_005.json
COPY data/cloudtrail_logs/development/cloudtrail_dev_001.json /workspace/cloudtrail_logs/development/cloudtrail_dev_001.json
COPY data/cloudtrail_logs/development/cloudtrail_dev_002.json /workspace/cloudtrail_logs/development/cloudtrail_dev_002.json
COPY data/cloudtrail_logs/development/cloudtrail_dev_003.json /workspace/cloudtrail_logs/development/cloudtrail_dev_003.json
COPY data/cloudtrail_logs/development/cloudtrail_dev_004.json /workspace/cloudtrail_logs/development/cloudtrail_dev_004.json
COPY data/cloudtrail_logs/development/cloudtrail_dev_005.json /workspace/cloudtrail_logs/development/cloudtrail_dev_005.json

WORKDIR /workspace