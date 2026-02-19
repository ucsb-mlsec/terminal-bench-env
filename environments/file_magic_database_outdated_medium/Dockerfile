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
    apt-get install -y file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/samples

COPY data/samples/test1.qvx /workspace/samples/test1.qvx
COPY data/samples/test2.qvx /workspace/samples/test2.qvx
COPY data/samples/test3.qvx /workspace/samples/test3.qvx

WORKDIR /workspace