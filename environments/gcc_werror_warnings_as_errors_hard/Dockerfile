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
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy_project/bin

COPY legacy_project/Makefile /workspace/legacy_project/Makefile
COPY legacy_project/main.c /workspace/legacy_project/main.c
COPY legacy_project/utils.c /workspace/legacy_project/utils.c
COPY legacy_project/parser.c /workspace/legacy_project/parser.c
COPY legacy_project/network.c /workspace/legacy_project/network.c
COPY legacy_project/database.c /workspace/legacy_project/database.c
COPY legacy_project/utils.h /workspace/legacy_project/utils.h
COPY legacy_project/parser.h /workspace/legacy_project/parser.h
COPY legacy_project/network.h /workspace/legacy_project/network.h
COPY legacy_project/database.h /workspace/legacy_project/database.h

WORKDIR /workspace/legacy_project