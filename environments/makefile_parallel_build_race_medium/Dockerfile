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
    make \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project

COPY project/Makefile /workspace/project/Makefile
COPY project/main.c /workspace/project/main.c
COPY project/utils.c /workspace/project/utils.c
COPY project/utils.h /workspace/project/utils.h
COPY project/processor.c /workspace/project/processor.c
COPY project/processor.h /workspace/project/processor.h
COPY project/handler.c /workspace/project/handler.c
COPY project/handler.h /workspace/project/handler.h
COPY project/generator.c /workspace/project/generator.c

WORKDIR /workspace/project