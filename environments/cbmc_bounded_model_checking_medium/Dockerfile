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
    apt-get install -y --no-install-recommends \
    gcc \
    make \
    cbmc && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/sensor_filter

COPY sensor_filter/filter.c /workspace/sensor_filter/filter.c
COPY sensor_filter/verify_config.txt /workspace/sensor_filter/verify_config.txt
COPY sensor_filter/README.md /workspace/sensor_filter/README.md
COPY sensor_filter/verify.sh /workspace/sensor_filter/verify.sh

RUN chmod +x /workspace/sensor_filter/verify.sh

WORKDIR /workspace