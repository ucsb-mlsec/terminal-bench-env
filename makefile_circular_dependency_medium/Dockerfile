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

RUN mkdir -p /workspace/dataproc /workspace/solution

COPY dataproc/main.c /workspace/dataproc/main.c
COPY dataproc/processor.c /workspace/dataproc/processor.c
COPY dataproc/utils.c /workspace/dataproc/utils.c
COPY dataproc/processor.h /workspace/dataproc/processor.h
COPY dataproc/utils.h /workspace/dataproc/utils.h
COPY dataproc/Makefile /workspace/dataproc/Makefile

WORKDIR /workspace