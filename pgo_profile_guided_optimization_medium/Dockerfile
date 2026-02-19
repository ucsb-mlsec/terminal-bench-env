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
    time && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc-dev time && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/text_analyzer/src && \
    mkdir -p /workspace/text_analyzer/data

COPY text_analyzer/src/analyzer.c /workspace/text_analyzer/src/analyzer.c
COPY text_analyzer/Makefile /workspace/text_analyzer/Makefile
COPY text_analyzer/data/sample1.txt /workspace/text_analyzer/data/sample1.txt
COPY text_analyzer/data/sample2.txt /workspace/text_analyzer/data/sample2.txt
COPY text_analyzer/data/sample3.txt /workspace/text_analyzer/data/sample3.txt
COPY text_analyzer/data/sample4.txt /workspace/text_analyzer/data/sample4.txt

WORKDIR /workspace