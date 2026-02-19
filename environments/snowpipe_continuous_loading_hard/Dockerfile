FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/incoming /workspace/data /workspace/solution

COPY data/incoming/events_001.json /workspace/incoming/events_001.json
COPY data/incoming/events_002.json /workspace/incoming/events_002.json
COPY data/incoming/events_003.json /workspace/incoming/events_003.json
COPY data/incoming/events_004.json /workspace/incoming/events_004.json
COPY data/incoming/events_005.json /workspace/incoming/events_005.json
COPY data/incoming/events_006.json /workspace/incoming/events_006.json
COPY data/incoming/events_007.json /workspace/incoming/events_007.json
COPY data/incoming/events_008.json /workspace/incoming/events_008.json
COPY data/incoming/events_009.json /workspace/incoming/events_009.json
COPY data/incoming/events_010.json /workspace/incoming/events_010.json

WORKDIR /workspace