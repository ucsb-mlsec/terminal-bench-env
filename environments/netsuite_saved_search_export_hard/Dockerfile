FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/data/netsuite_fragments

COPY data/netsuite_fragments/page_001.json /var/data/netsuite_fragments/page_001.json
COPY data/netsuite_fragments/page_002.json /var/data/netsuite_fragments/page_002.json
COPY data/netsuite_fragments/page_004.json /var/data/netsuite_fragments/page_004.json
COPY data/netsuite_fragments/page_005.json /var/data/netsuite_fragments/page_005.json
COPY data/netsuite_fragments/page_006.json /var/data/netsuite_fragments/page_006.json
COPY data/netsuite_fragments/page_007.json /var/data/netsuite_fragments/page_007.json
COPY data/netsuite_fragments/page_008.json /var/data/netsuite_fragments/page_008.json

WORKDIR /var/data/netsuite_fragments