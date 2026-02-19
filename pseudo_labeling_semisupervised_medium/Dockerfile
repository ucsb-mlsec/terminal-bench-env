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
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy scikit-learn pillow

RUN pip3 install --no-cache-dir --break-system-packages \
    scikit-learn \
    numpy \
    pillow

RUN mkdir -p /data/labeled/electronics /data/labeled/clothing /data/labeled/furniture \
    /data/unlabeled \
    /data/test/electronics /data/test/clothing /data/test/furniture \
    /solution

COPY data/labeled/electronics/img_001.png /data/labeled/electronics/img_001.png
COPY data/labeled/clothing/img_001.png /data/labeled/clothing/img_001.png
COPY data/labeled/furniture/img_001.png /data/labeled/furniture/img_001.png
COPY data/unlabeled/img_u001.png /data/unlabeled/img_u001.png
COPY data/test/electronics/test_001.png /data/test/electronics/test_001.png
COPY data/test/clothing/test_001.png /data/test/clothing/test_001.png
COPY data/test/furniture/test_001.png /data/test/furniture/test_001.png

WORKDIR /workspace