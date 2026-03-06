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
    apt-get install -y build-essential gcc make valgrind && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/config_parser /workspace/test_configs

COPY config_parser/parser.c /workspace/config_parser/parser.c
COPY config_parser/Makefile /workspace/config_parser/Makefile
COPY data/test_configs/config1.txt /workspace/test_configs/config1.txt
COPY data/test_configs/config2.txt /workspace/test_configs/config2.txt
COPY data/test_configs/config3.txt /workspace/test_configs/config3.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
