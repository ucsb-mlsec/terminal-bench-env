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
    apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/worker/payloads

COPY processor.js /opt/worker/processor.js
COPY data/small.json /opt/worker/payloads/small.json
COPY data/medium.json /opt/worker/payloads/medium.json
COPY data/large.json /opt/worker/payloads/large.json
COPY measure.js /opt/worker/measure.js

WORKDIR /opt/worker