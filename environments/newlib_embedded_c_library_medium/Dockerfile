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
    libnewlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make binutils libc6-dev libnewlib-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/libutils /workspace/test_app

COPY libutils/string_utils.c /workspace/libutils/string_utils.c
COPY libutils/math_utils.c /workspace/libutils/math_utils.c
COPY libutils/array_utils.c /workspace/libutils/array_utils.c
COPY libutils/string_utils.h /workspace/libutils/string_utils.h
COPY libutils/math_utils.h /workspace/libutils/math_utils.h
COPY libutils/array_utils.h /workspace/libutils/array_utils.h
COPY test_app/main.c /workspace/test_app/main.c

WORKDIR /workspace