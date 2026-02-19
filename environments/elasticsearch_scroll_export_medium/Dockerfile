FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/scroll_data /tmp

COPY data/scroll_data/page_001.json /tmp/scroll_data/page_001.json
COPY data/scroll_data/page_002.json /tmp/scroll_data/page_002.json
COPY data/scroll_data/page_003.json /tmp/scroll_data/page_003.json
COPY data/scroll_data/page_004.json /tmp/scroll_data/page_004.json
COPY data/scroll_data/page_005.json /tmp/scroll_data/page_005.json

WORKDIR /tmp