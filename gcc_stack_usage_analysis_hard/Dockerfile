FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/firmware/config /workspace/firmware/data

COPY config/stack_config.txt /workspace/firmware/stack_config.txt
COPY data/module1.su /workspace/firmware/module1.su
COPY data/module2.su /workspace/firmware/module2.su
COPY data/handlers.su /workspace/firmware/handlers.su
COPY data/protocol.su /workspace/firmware/protocol.su
COPY data/utilities.su /workspace/firmware/utilities.su

WORKDIR /workspace