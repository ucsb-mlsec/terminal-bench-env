FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/graphs

COPY data/graph_0.txt /workspace/graphs/graph_0.txt
COPY data/graph_1.txt /workspace/graphs/graph_1.txt
COPY data/graph_2.txt /workspace/graphs/graph_2.txt
COPY data/graph_3.txt /workspace/graphs/graph_3.txt
COPY data/graph_4.txt /workspace/graphs/graph_4.txt
COPY data/graph_5.txt /workspace/graphs/graph_5.txt
COPY data/graph_6.txt /workspace/graphs/graph_6.txt
COPY data/graph_7.txt /workspace/graphs/graph_7.txt

WORKDIR /workspace