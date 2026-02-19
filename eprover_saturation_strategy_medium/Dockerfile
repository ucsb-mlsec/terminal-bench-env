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
    apt-get install -y eprover && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/configs /workspace/benchmarks

COPY config/default.txt /workspace/configs/default.txt
COPY config/optimized.txt /workspace/configs/optimized.txt
COPY data/problem1.p /workspace/benchmarks/problem1.p
COPY data/problem2.p /workspace/benchmarks/problem2.p
COPY data/problem3.p /workspace/benchmarks/problem3.p
COPY data/problem4.p /workspace/benchmarks/problem4.p
COPY data/problem5.p /workspace/benchmarks/problem5.p

WORKDIR /workspace