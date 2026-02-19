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

RUN pip3 install --no-cache-dir --break-system-packages torch torchvision numpy pillow

RUN mkdir -p /workspace/data

COPY train_model.py /workspace/train_model.py
COPY data/img_001.png /workspace/data/img_001.png
COPY data/img_002.png /workspace/data/img_002.png
COPY data/img_003.png /workspace/data/img_003.png
COPY data/img_004.png /workspace/data/img_004.png
COPY data/img_005.png /workspace/data/img_005.png
COPY data/placeholder.txt /workspace/data/placeholder.txt

WORKDIR /workspace