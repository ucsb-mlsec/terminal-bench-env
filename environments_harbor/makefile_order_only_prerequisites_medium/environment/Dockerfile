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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/output/makefile_order_only_prerequisites_medium/files/src /workspace/output/makefile_order_only_prerequisites_medium/files/include

COPY src/main.c /workspace/output/makefile_order_only_prerequisites_medium/files/src/main.c
COPY src/utils.c /workspace/output/makefile_order_only_prerequisites_medium/files/src/utils.c
COPY include/utils.h /workspace/output/makefile_order_only_prerequisites_medium/files/include/utils.h
COPY Makefile /workspace/output/makefile_order_only_prerequisites_medium/files/Makefile

WORKDIR /workspace/output/makefile_order_only_prerequisites_medium/files

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
