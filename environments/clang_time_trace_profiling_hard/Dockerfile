FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/traces

COPY data/traces/file1.json /workspace/traces/file1.json
COPY data/traces/file2.json /workspace/traces/file2.json
COPY data/traces/file3.json /workspace/traces/file3.json
COPY data/traces/file4.json /workspace/traces/file4.json
COPY data/traces/file5.json /workspace/traces/file5.json

WORKDIR /workspace