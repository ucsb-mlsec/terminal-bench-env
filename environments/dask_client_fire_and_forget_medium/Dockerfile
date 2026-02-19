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
    apt-get install -y --no-install-recommends \
    python3-pip \
    gzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages dask[complete] distributed

RUN mkdir -p /workspace/logs/active /workspace/logs/archive /workspace/solution

COPY data/logs/active/app1.log /workspace/logs/active/app1.log
COPY data/logs/active/app2.log /workspace/logs/active/app2.log
COPY data/logs/active/app3.log /workspace/logs/active/app3.log
COPY data/logs/active/service.log /workspace/logs/active/service.log
COPY data/logs/active/monitor.log /workspace/logs/active/monitor.log

WORKDIR /workspace