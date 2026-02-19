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
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision pillow

RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision \
    pillow \
    numpy

RUN mkdir -p /workspace/dataset /workspace/config

COPY train.py /workspace/train.py
COPY config/config.json /workspace/config.json
COPY data/dataset/lr_image_1.png /workspace/dataset/lr_image_1.png
COPY data/dataset/hr_image_1.png /workspace/dataset/hr_image_1.png
COPY data/dataset/lr_image_2.png /workspace/dataset/lr_image_2.png
COPY data/dataset/hr_image_2.png /workspace/dataset/hr_image_2.png
COPY data/dataset/lr_image_3.png /workspace/dataset/lr_image_3.png
COPY data/dataset/hr_image_3.png /workspace/dataset/hr_image_3.png
COPY data/dataset/lr_image_4.png /workspace/dataset/lr_image_4.png
COPY data/dataset/hr_image_4.png /workspace/dataset/hr_image_4.png
COPY data/dataset/lr_image_5.png /workspace/dataset/lr_image_5.png
COPY data/dataset/hr_image_5.png /workspace/dataset/hr_image_5.png

WORKDIR /workspace