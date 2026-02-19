FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/examples

COPY data/simclr_config.json /workspace/simclr_config.json
COPY data/spec.txt /workspace/spec.txt
COPY data/examples/example1.json /workspace/examples/example1.json
COPY data/examples/example2.json /workspace/examples/example2.json
COPY data/examples/example3.json /workspace/examples/example3.json

WORKDIR /workspace