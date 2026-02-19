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
    mingw-w64 \
    gcc-mingw-w64-x86-64 \
    g++-mingw-w64-x86-64 \
    binutils-mingw-w64-x86-64 \
    make \
    file && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 binutils-mingw-w64-x86-64 mingw-w64 make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/filestat/build /workspace/data

COPY filestat/main.c /workspace/filestat/main.c
COPY filestat/processor.c /workspace/filestat/processor.c
COPY filestat/processor.h /workspace/filestat/processor.h
COPY filestat/Makefile /workspace/filestat/Makefile
COPY data/test_input.txt /workspace/filestat/test_input.txt

WORKDIR /workspace