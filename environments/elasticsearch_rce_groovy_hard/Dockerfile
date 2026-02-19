FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/elasticsearch_capture

COPY data/captured_payloads.json /tmp/elasticsearch_capture/captured_payloads.json
COPY data/index_schema.json /tmp/elasticsearch_capture/index_schema.json
COPY data/index_data.json /tmp/elasticsearch_capture/index_data.json

WORKDIR /workspace