FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data

COPY processor.py /workspace/processor.py
COPY data/data_0.json /workspace/data/data_0.json
COPY data/data_1.json /workspace/data/data_1.json
COPY data/data_2.json /workspace/data/data_2.json
COPY data/data_3.json /workspace/data/data_3.json
COPY data/data_4.json /workspace/data/data_4.json
COPY data/data_5.json /workspace/data/data_5.json
COPY data/data_6.json /workspace/data/data_6.json
COPY data/data_7.json /workspace/data/data_7.json
COPY data/data_9.json /workspace/data/data_9.json

WORKDIR /workspace