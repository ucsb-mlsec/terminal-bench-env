FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/pipeline/test_data

COPY pipeline/processor.py /workspace/pipeline/processor.py
COPY pipeline/test_data/frame_001.txt /workspace/pipeline/test_data/frame_001.txt
COPY pipeline/test_data/frame_002.txt /workspace/pipeline/test_data/frame_002.txt
COPY pipeline/test_data/frame_003.txt /workspace/pipeline/test_data/frame_003.txt
COPY pipeline/test_data/frame_004.txt /workspace/pipeline/test_data/frame_004.txt
COPY pipeline/test_data/frame_005.txt /workspace/pipeline/test_data/frame_005.txt

WORKDIR /workspace