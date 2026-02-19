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
    apt-get install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/textutil/parser \
    /workspace/textutil/formatter \
    /workspace/textutil/output \
    /workspace/textutil/include

COPY workspace/textutil/parser/parser.c /workspace/textutil/parser/parser.c
COPY workspace/textutil/parser/Makefile /workspace/textutil/parser/Makefile
COPY workspace/textutil/formatter/formatter.c /workspace/textutil/formatter/formatter.c
COPY workspace/textutil/formatter/Makefile /workspace/textutil/formatter/Makefile
COPY workspace/textutil/output/output.c /workspace/textutil/output/output.c
COPY workspace/textutil/output/Makefile /workspace/textutil/output/Makefile
COPY workspace/textutil/include/parser.h /workspace/textutil/include/parser.h
COPY workspace/textutil/include/formatter.h /workspace/textutil/include/formatter.h
COPY workspace/textutil/include/output.h /workspace/textutil/include/output.h
COPY workspace/textutil/include/common.h /workspace/textutil/include/common.h
COPY workspace/textutil/main.c /workspace/textutil/main.c
COPY workspace/textutil/Makefile /workspace/textutil/Makefile

WORKDIR /workspace/textutil