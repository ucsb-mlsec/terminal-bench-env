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
    apt-get install -y build-essential gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib

COPY mathlib/add.c /workspace/mathlib/add.c
COPY mathlib/add.h /workspace/mathlib/add.h
COPY mathlib/subtract.c /workspace/mathlib/subtract.c
COPY mathlib/subtract.h /workspace/mathlib/subtract.h
COPY mathlib/multiply.c /workspace/mathlib/multiply.c
COPY mathlib/multiply.h /workspace/mathlib/multiply.h
COPY mathlib/divide.c /workspace/mathlib/divide.c
COPY mathlib/divide.h /workspace/mathlib/divide.h
COPY mathlib/test_math.c /workspace/mathlib/test_math.c
COPY mathlib/build.sh /workspace/mathlib/build.sh

RUN chmod +x /workspace/mathlib/build.sh

WORKDIR /workspace/mathlib