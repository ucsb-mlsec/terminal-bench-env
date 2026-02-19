FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/kinesis /tmp

COPY data/traffic_data.json /home/user/kinesis/traffic_data.json
COPY data/AWS_KINESIS_SPECS.md /home/user/kinesis/AWS_KINESIS_SPECS.md
COPY shard_calculator.py /home/user/kinesis/shard_calculator.py

WORKDIR /home/user/kinesis