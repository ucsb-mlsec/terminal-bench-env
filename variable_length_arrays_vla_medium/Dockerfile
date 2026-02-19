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
    apt-get install -y gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/image_processor /workspace/test_images

COPY image_processor/processor.c /workspace/image_processor/processor.c
COPY data/test_images/small.txt /workspace/test_images/small.txt
COPY data/test_images/medium.txt /workspace/test_images/medium.txt
COPY data/test_images/large.txt /workspace/test_images/large.txt
COPY data/test_images/xlarge.txt /workspace/test_images/xlarge.txt

WORKDIR /workspace