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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages Pillow

RUN mkdir -p /workspace/sd_dataset/images

COPY data/sd_dataset/images/image_001.jpg /workspace/sd_dataset/images/image_001.jpg
COPY data/sd_dataset/images/image_002.png /workspace/sd_dataset/images/image_002.png
COPY data/sd_dataset/images/image_003.jpg /workspace/sd_dataset/images/image_003.jpg
COPY data/sd_dataset/images/image_004.webp /workspace/sd_dataset/images/image_004.webp
COPY data/sd_dataset/images/image_005.jpg /workspace/sd_dataset/images/image_005.jpg
COPY data/sd_dataset/images/image_006.png /workspace/sd_dataset/images/image_006.png
COPY data/sd_dataset/images/image_007.jpg /workspace/sd_dataset/images/image_007.jpg
COPY data/sd_dataset/images/image_008.jpg /workspace/sd_dataset/images/image_008.jpg
COPY data/sd_dataset/images/image_009.jpg /workspace/sd_dataset/images/image_009.jpg
COPY data/sd_dataset/images/image_010.png /workspace/sd_dataset/images/image_010.png
COPY data/sd_dataset/images/image_011.jpg /workspace/sd_dataset/images/image_011.jpg
COPY data/sd_dataset/images/image_012.jpg /workspace/sd_dataset/images/image_012.jpg
COPY data/sd_dataset/captions.txt /workspace/sd_dataset/captions.txt
COPY data/sd_dataset/metadata.json /workspace/sd_dataset/metadata.json

WORKDIR /workspace