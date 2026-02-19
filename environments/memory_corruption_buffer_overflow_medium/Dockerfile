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
    apt-get install -y \
    gcc \
    make \
    gdb \
    valgrind && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/logs

COPY logparser.c /workspace/logparser.c
COPY Makefile /workspace/Makefile
COPY data/test1.log /workspace/logs/test1.log
COPY data/test2.log /workspace/logs/test2.log
COPY data/test3.log /workspace/logs/test3.log
COPY data/test4.log /workspace/logs/test4.log
COPY data/test5.log /workspace/logs/test5.log

WORKDIR /workspace