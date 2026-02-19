FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config /workspace/data

COPY config/current_config.conf /workspace/current_config.conf
COPY data/error_log.txt /workspace/error_log.txt
COPY data/object_sizes.json /workspace/object_sizes.json
COPY data/performance_constraints.txt /workspace/performance_constraints.txt

WORKDIR /workspace