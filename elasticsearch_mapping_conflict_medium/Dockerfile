FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/product_data

COPY data/product_data/product_001.json /home/user/product_data/product_001.json
COPY data/product_data/product_002.json /home/user/product_data/product_002.json
COPY data/product_data/product_003.json /home/user/product_data/product_003.json
COPY data/product_data/product_004.json /home/user/product_data/product_004.json
COPY data/product_data/product_005.json /home/user/product_data/product_005.json
COPY data/product_data/product_006.json /home/user/product_data/product_006.json
COPY data/product_data/product_007.json /home/user/product_data/product_007.json
COPY data/product_data/product_008.json /home/user/product_data/product_008.json
COPY data/product_data/product_009.json /home/user/product_data/product_009.json
COPY data/product_data/product_010.json /home/user/product_data/product_010.json
COPY data/product_data/product_011.json /home/user/product_data/product_011.json
COPY data/product_data/product_012.json /home/user/product_data/product_012.json

WORKDIR /home/user