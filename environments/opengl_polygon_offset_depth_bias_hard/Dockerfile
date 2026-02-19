FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/depth_analysis /solution

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy scipy

COPY data/test_results.csv /workspace/depth_analysis/test_results.csv
COPY data/depth_precision_specs.txt /workspace/depth_analysis/depth_precision_specs.txt
COPY data/README.txt /workspace/depth_analysis/README.txt

WORKDIR /workspace