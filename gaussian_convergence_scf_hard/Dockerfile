FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/gaussian_logs

COPY data/gaussian_logs/calc_001.log /workspace/gaussian_logs/calc_001.log
COPY data/gaussian_logs/calc_002.log /workspace/gaussian_logs/calc_002.log
COPY data/gaussian_logs/calc_003.log /workspace/gaussian_logs/calc_003.log
COPY data/gaussian_logs/calc_004.log /workspace/gaussian_logs/calc_004.log
COPY data/gaussian_logs/calc_005.log /workspace/gaussian_logs/calc_005.log
COPY data/gaussian_logs/calc_006.log /workspace/gaussian_logs/calc_006.log
COPY data/gaussian_logs/calc_007.log /workspace/gaussian_logs/calc_007.log
COPY data/gaussian_logs/calc_008.log /workspace/gaussian_logs/calc_008.log
COPY data/gaussian_logs/calc_009.log /workspace/gaussian_logs/calc_009.log
COPY data/gaussian_logs/calc_010.log /workspace/gaussian_logs/calc_010.log
COPY data/gaussian_logs/calc_011.log /workspace/gaussian_logs/calc_011.log
COPY data/gaussian_logs/calc_012.log /workspace/gaussian_logs/calc_012.log
COPY data/gaussian_logs/calc_013.log /workspace/gaussian_logs/calc_013.log
COPY data/gaussian_logs/calc_014.log /workspace/gaussian_logs/calc_014.log
COPY data/gaussian_logs/calc_015.log /workspace/gaussian_logs/calc_015.log

WORKDIR /workspace