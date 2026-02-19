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
    valgrind && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make valgrind && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/log_analyzer/logs

COPY log_parser.h /workspace/log_analyzer/
COPY log_parser.c /workspace/log_analyzer/
COPY stats.h /workspace/log_analyzer/
COPY stats.c /workspace/log_analyzer/
COPY Makefile /workspace/log_analyzer/
COPY data/test1.log /workspace/log_analyzer/logs/
COPY data/test2.log /workspace/log_analyzer/logs/
COPY data/test3.log /workspace/log_analyzer/logs/
COPY data/test4.log /workspace/log_analyzer/logs/
COPY run_analysis.sh /workspace/log_analyzer/

RUN chmod +x /workspace/log_analyzer/run_analysis.sh

WORKDIR /workspace/log_analyzer