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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN mkdir -p /workspace/processor /workspace/inputs

COPY transform.py /workspace/processor/transform.py
COPY data/data_001.json /workspace/inputs/data_001.json
COPY data/data_002.json /workspace/inputs/data_002.json
COPY data/data_003.json /workspace/inputs/data_003.json
COPY data/data_004.json /workspace/inputs/data_004.json
COPY data/data_005.json /workspace/inputs/data_005.json

WORKDIR /workspace