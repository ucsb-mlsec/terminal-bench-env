FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/configs /workspace/fixed

COPY data/configs/server1.json /workspace/configs/server1.json
COPY data/configs/server2.json /workspace/configs/server2.json
COPY data/configs/server3.json /workspace/configs/server3.json
COPY data/configs/server4.json /workspace/configs/server4.json
COPY data/configs/server5.json /workspace/configs/server5.json
COPY data/configs/server6.json /workspace/configs/server6.json
COPY data/configs/server8.json /workspace/configs/server8.json

WORKDIR /workspace