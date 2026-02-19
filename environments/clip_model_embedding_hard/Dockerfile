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
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages torch transformers pillow

RUN pip3 install --no-cache-dir --break-system-packages \
    torch \
    torchvision \
    transformers \
    Pillow

RUN mkdir -p /workspace/test_images

COPY data/image_metadata.json /workspace/image_metadata.json
COPY data/test_images/image_00.png /workspace/test_images/image_00.png
COPY data/test_images/image_01.png /workspace/test_images/image_01.png
COPY data/test_images/image_02.png /workspace/test_images/image_02.png
COPY data/test_images/image_03.png /workspace/test_images/image_03.png
COPY data/test_images/image_04.png /workspace/test_images/image_04.png
COPY data/test_images/image_05.png /workspace/test_images/image_05.png
COPY data/test_images/image_06.png /workspace/test_images/image_06.png
COPY data/test_images/image_07.png /workspace/test_images/image_07.png
COPY data/test_images/image_08.png /workspace/test_images/image_08.png
COPY data/test_images/image_09.png /workspace/test_images/image_09.png

WORKDIR /workspace