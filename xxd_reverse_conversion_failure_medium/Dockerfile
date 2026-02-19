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
    apt-get install -y xxd vim-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/hex_dumps

COPY data/hex_dumps/backup1.hex /workspace/hex_dumps/backup1.hex
COPY data/hex_dumps/backup2.hex /workspace/hex_dumps/backup2.hex
COPY data/hex_dumps/backup3.hex /workspace/hex_dumps/backup3.hex
COPY data/hex_dumps/backup4.hex /workspace/hex_dumps/backup4.hex
COPY data/hex_dumps/backup5.hex /workspace/hex_dumps/backup5.hex
COPY data/hex_dumps/backup6.hex /workspace/hex_dumps/backup6.hex

WORKDIR /workspace