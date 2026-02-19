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
    g++ \
    make \
    libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/src /workspace/project/config /solution

COPY workspace/project/src/matrix_multiply.c /workspace/project/src/matrix_multiply.c
COPY workspace/project/src/vector_operations.cpp /workspace/project/src/vector_operations.cpp
COPY workspace/project/src/reduction_ops.c /workspace/project/src/reduction_ops.c
COPY workspace/project/src/utils.cpp /workspace/project/src/utils.cpp
COPY workspace/project/Makefile /workspace/project/Makefile
COPY workspace/project/config/runtime.conf /workspace/project/config/runtime.conf
COPY workspace/project/config/app_settings.ini /workspace/project/config/app_settings.ini

WORKDIR /workspace