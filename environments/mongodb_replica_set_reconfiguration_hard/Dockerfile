FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/mongodb/status_dumps /tmp

COPY data/node1_status.json /var/mongodb/status_dumps/node1_status.json
COPY data/node2_status.json /var/mongodb/status_dumps/node2_status.json
COPY data/node3_status.json /var/mongodb/status_dumps/node3_status.json
COPY data/node4_status.json /var/mongodb/status_dumps/node4_status.json
COPY data/node5_status.json /var/mongodb/status_dumps/node5_status.json

WORKDIR /workspace