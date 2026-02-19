FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/app /workspace/data

COPY data/valgrind_output.txt /workspace/valgrind_output.txt
COPY app/main.c /workspace/app/main.c
COPY app/data_processor.c /workspace/app/data_processor.c
COPY app/utils.c /workspace/app/utils.c
COPY app/logger.c /workspace/app/logger.c

WORKDIR /workspace