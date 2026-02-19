FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/query_stats

COPY data/query_stats/query_1.stats /workspace/query_stats/query_1.stats
COPY data/query_stats/query_2.stats /workspace/query_stats/query_2.stats
COPY data/query_stats/query_3.stats /workspace/query_stats/query_3.stats
COPY data/query_stats/query_4.stats /workspace/query_stats/query_4.stats
COPY data/query_stats/query_5.stats /workspace/query_stats/query_5.stats
COPY data/query_stats/query_6.stats /workspace/query_stats/query_6.stats

WORKDIR /workspace