FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/methylation_calls /output

COPY data/methylation_calls/sample_001.txt /data/methylation_calls/sample_001.txt
COPY data/methylation_calls/sample_002.txt /data/methylation_calls/sample_002.txt
COPY data/methylation_calls/sample_003.txt /data/methylation_calls/sample_003.txt
COPY data/methylation_calls/sample_004.txt /data/methylation_calls/sample_004.txt
COPY data/methylation_calls/sample_005.txt /data/methylation_calls/sample_005.txt
COPY data/methylation_calls/sample_006.txt /data/methylation_calls/sample_006.txt

WORKDIR /workspace