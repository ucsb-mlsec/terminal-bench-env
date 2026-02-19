FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/network_data /workspace

COPY data/network_data/region1.txt /workspace/network_data/region1.txt
COPY data/network_data/region2.txt /workspace/network_data/region2.txt
COPY data/network_data/region3.txt /workspace/network_data/region3.txt
COPY data/network_data/inter_region_links.txt /workspace/network_data/inter_region_links.txt
COPY data/network_data/region4_isolated.txt /workspace/network_data/region4_isolated.txt
COPY data/network_data/additional_connections.conf /workspace/network_data/additional_connections.conf

WORKDIR /workspace