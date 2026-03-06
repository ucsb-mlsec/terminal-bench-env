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
    apt-get install -y build-essential gcc make libomp-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/histogram /workspace/histogram/data

COPY histogram.c /workspace/histogram/
COPY Makefile /workspace/histogram/
COPY data/data1.txt /workspace/histogram/data/
COPY data/data2.txt /workspace/histogram/data/
COPY data/data3.txt /workspace/histogram/data/
COPY data/data4.txt /workspace/histogram/data/
COPY data/data5.txt /workspace/histogram/data/
COPY data/data6.txt /workspace/histogram/data/
COPY data/data7.txt /workspace/histogram/data/
COPY data/data8.txt /workspace/histogram/data/
COPY data/data9.txt /workspace/histogram/data/
COPY data/data10.txt /workspace/histogram/data/
COPY reference_output.txt /workspace/histogram/

WORKDIR /workspace/histogram

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
