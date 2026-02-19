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

RUN pip3 install --no-cache-dir --break-system-packages Pillow

RUN mkdir -p /workspace/photos

COPY data/photos/image001.jpg /workspace/photos/image001.jpg
COPY data/photos/image002.jpg /workspace/photos/image002.jpg
COPY data/photos/image003.jpg /workspace/photos/image003.jpg
COPY data/photos/image004.jpg /workspace/photos/image004.jpg
COPY data/photos/image005.jpg /workspace/photos/image005.jpg
COPY data/photos/image006.jpg /workspace/photos/image006.jpg
COPY data/photos/image007.jpg /workspace/photos/image007.jpg
COPY data/photos/image008.jpg /workspace/photos/image008.jpg

WORKDIR /workspace