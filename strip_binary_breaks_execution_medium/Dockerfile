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
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/processor/src \
    /workspace/processor/bin \
    /workspace/processor

COPY processor/src/main.c /workspace/processor/src/main.c
COPY processor/src/validator.c /workspace/processor/src/validator.c
COPY processor/src/validator.h /workspace/processor/src/validator.h
COPY processor/Makefile /workspace/processor/Makefile
COPY data/test_data.csv /workspace/processor/test_data.csv
COPY processor/bin/dataproc /workspace/processor/bin/dataproc

WORKDIR /workspace