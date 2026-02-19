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
    apt-get install -y build-essential make gcc && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/stats-tool

COPY analyzer.c /workspace/stats-tool/analyzer.c
COPY Makefile /workspace/stats-tool/Makefile
COPY data/data.csv /workspace/stats-tool/data.csv

WORKDIR /workspace/stats-tool