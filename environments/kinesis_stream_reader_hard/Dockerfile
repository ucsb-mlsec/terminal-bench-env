FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/stream /home/user

COPY data/stream/batch_0001.json /data/stream/batch_0001.json
COPY data/stream/batch_0002.json /data/stream/batch_0002.json
COPY data/stream/batch_0003.json /data/stream/batch_0003.json
COPY data/stream/batch_0004.json /data/stream/batch_0004.json
COPY data/stream/batch_0006.json /data/stream/batch_0006.json
COPY data/stream/batch_0007.json /data/stream/batch_0007.json
COPY data/stream/batch_0008.json /data/stream/batch_0008.json

WORKDIR /home/user