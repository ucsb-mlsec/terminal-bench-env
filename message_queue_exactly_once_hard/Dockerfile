FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/queue /workspace/completed /workspace

COPY processor.py /workspace/processor.py
COPY data/queue/payment_001.json /workspace/queue/payment_001.json
COPY data/queue/payment_002.json /workspace/queue/payment_002.json
COPY data/queue/payment_003.json /workspace/queue/payment_003.json
COPY data/queue/payment_005.json /workspace/queue/payment_005.json
COPY data/queue/payment_006.json /workspace/queue/payment_006.json
COPY data/queue/payment_007.json /workspace/queue/payment_007.json
COPY data/queue/payment_008.json /workspace/queue/payment_008.json
COPY data/queue/payment_009.json /workspace/queue/payment_009.json
COPY data/state.json /workspace/state.json

WORKDIR /workspace