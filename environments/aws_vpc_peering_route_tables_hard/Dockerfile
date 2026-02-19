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
    python3-pip \
    jq && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages boto3

RUN mkdir -p /workspace/aws-data

COPY data/vpcs.json /workspace/aws-data/vpcs.json
COPY data/peering_connections.json /workspace/aws-data/peering_connections.json
COPY data/route_tables.json /workspace/aws-data/route_tables.json
COPY data/subnets.json /workspace/aws-data/subnets.json
COPY data/aws_config.json /workspace/aws-data/aws_config.json

WORKDIR /workspace