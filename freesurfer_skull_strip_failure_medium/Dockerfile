FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/scan_metadata /solution

COPY data/scan_metadata/scan_001.txt /data/scan_metadata/scan_001.txt
COPY data/scan_metadata/scan_002.txt /data/scan_metadata/scan_002.txt
COPY data/scan_metadata/scan_003.txt /data/scan_metadata/scan_003.txt
COPY data/scan_metadata/scan_004.txt /data/scan_metadata/scan_004.txt
COPY data/scan_metadata/scan_005.txt /data/scan_metadata/scan_005.txt
COPY data/scan_metadata/scan_006.txt /data/scan_metadata/scan_006.txt
COPY data/scan_metadata/scan_007.txt /data/scan_metadata/scan_007.txt
COPY data/scan_metadata/scan_008.txt /data/scan_metadata/scan_008.txt
COPY data/scan_metadata/scan_009.txt /data/scan_metadata/scan_009.txt
COPY data/scan_metadata/scan_010.txt /data/scan_metadata/scan_010.txt
COPY data/scan_metadata/scan_011.txt /data/scan_metadata/scan_011.txt
COPY data/scan_metadata/scan_012.txt /data/scan_metadata/scan_012.txt
COPY data/scan_metadata/scan_013.txt /data/scan_metadata/scan_013.txt
COPY data/scan_metadata/scan_014.txt /data/scan_metadata/scan_014.txt
COPY data/scan_metadata/scan_015.txt /data/scan_metadata/scan_015.txt
COPY data/scan_metadata/scan_016.txt /data/scan_metadata/scan_016.txt
COPY data/scan_metadata/scan_017.txt /data/scan_metadata/scan_017.txt
COPY data/scan_metadata/scan_018.txt /data/scan_metadata/scan_018.txt
COPY data/scan_metadata/scan_019.txt /data/scan_metadata/scan_019.txt
COPY data/scan_metadata/scan_020.txt /data/scan_metadata/scan_020.txt

WORKDIR /workspace