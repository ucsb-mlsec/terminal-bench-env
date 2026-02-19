FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/circuits

COPY data/circuits/adder.blif /workspace/circuits/adder.blif
COPY data/circuits/multiplexer.blif /workspace/circuits/multiplexer.blif
COPY data/circuits/alu.blif /workspace/circuits/alu.blif
COPY data/circuits/decoder.blif /workspace/circuits/decoder.blif
COPY data/circuits/comparator.blif /workspace/circuits/comparator.blif
COPY data/circuits/encoder.blif /workspace/circuits/encoder.blif

WORKDIR /workspace