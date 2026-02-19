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
    g++ \
    make \
    mold && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential mold && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/dataproc

COPY workspace/dataproc/main.cpp /workspace/dataproc/main.cpp
COPY workspace/dataproc/parser.cpp /workspace/dataproc/parser.cpp
COPY workspace/dataproc/filter.cpp /workspace/dataproc/filter.cpp
COPY workspace/dataproc/aggregator.cpp /workspace/dataproc/aggregator.cpp
COPY workspace/dataproc/transformer.cpp /workspace/dataproc/transformer.cpp
COPY workspace/dataproc/validator.cpp /workspace/dataproc/validator.cpp
COPY workspace/dataproc/writer.cpp /workspace/dataproc/writer.cpp
COPY workspace/dataproc/config.cpp /workspace/dataproc/config.cpp
COPY workspace/dataproc/parser.h /workspace/dataproc/parser.h
COPY workspace/dataproc/filter.h /workspace/dataproc/filter.h
COPY workspace/dataproc/aggregator.h /workspace/dataproc/aggregator.h
COPY workspace/dataproc/transformer.h /workspace/dataproc/transformer.h
COPY workspace/dataproc/validator.h /workspace/dataproc/validator.h
COPY workspace/dataproc/writer.h /workspace/dataproc/writer.h
COPY workspace/dataproc/config.h /workspace/dataproc/config.h
COPY workspace/dataproc/Makefile /workspace/dataproc/Makefile

WORKDIR /workspace/dataproc