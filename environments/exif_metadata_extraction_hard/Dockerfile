FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /evidence/metadata /solution

COPY data/metadata/image_001.txt /evidence/metadata/image_001.txt
COPY data/metadata/image_002.txt /evidence/metadata/image_002.txt
COPY data/metadata/image_003.txt /evidence/metadata/image_003.txt
COPY data/metadata/image_004.txt /evidence/metadata/image_004.txt
COPY data/metadata/image_005.txt /evidence/metadata/image_005.txt
COPY data/metadata/image_006.txt /evidence/metadata/image_006.txt
COPY data/metadata/image_007.txt /evidence/metadata/image_007.txt
COPY data/metadata/image_008.txt /evidence/metadata/image_008.txt

WORKDIR /workspace