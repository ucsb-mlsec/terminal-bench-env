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
    build-essential \
    gcc \
    make \
    gdb \
    valgrind && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/transaction_service

COPY transaction_service/transaction_service.c /workspace/transaction_service/transaction_service.c
COPY transaction_service/account.c /workspace/transaction_service/account.c
COPY transaction_service/account.h /workspace/transaction_service/account.h
COPY transaction_service/Makefile /workspace/transaction_service/Makefile
COPY transaction_service/test_harness.sh /workspace/transaction_service/test_harness.sh

RUN chmod +x /workspace/transaction_service/test_harness.sh

WORKDIR /workspace