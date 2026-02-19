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

RUN pip3 install --no-cache-dir --break-system-packages numpy pillow opencv-python

RUN mkdir -p /workspace/input /workspace/output

COPY data/input/img1.png /workspace/input/img1.png
COPY data/input/img2.png /workspace/input/img2.png
COPY data/input/img3.png /workspace/input/img3.png
COPY data/input/img4.png /workspace/input/img4.png
COPY data/input/img5.png /workspace/input/img5.png

WORKDIR /workspace