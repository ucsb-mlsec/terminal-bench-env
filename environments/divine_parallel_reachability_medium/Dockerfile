FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data /workspace/explorer /workspace/solution

COPY data/state_graph.json /workspace/data/state_graph.json
COPY explorer/parallel_explore.py /workspace/explorer/parallel_explore.py
COPY test_runner.sh /workspace/test_runner.sh

RUN chmod +x /workspace/test_runner.sh

WORKDIR /workspace