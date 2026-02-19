FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/test_data /workspace/solution

COPY data/small.txt /workspace/test_data/small.txt
COPY data/medium.txt /workspace/test_data/medium.txt
COPY data/large.txt /workspace/test_data/large.txt
COPY data/binary.dat /workspace/test_data/binary.dat

WORKDIR /workspace