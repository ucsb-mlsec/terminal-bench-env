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
    distcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y distcc gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp_project /root/solution

COPY workspace/cpp_project/main.cpp /workspace/cpp_project/main.cpp
COPY workspace/cpp_project/utils.cpp /workspace/cpp_project/utils.cpp
COPY workspace/cpp_project/processing.cpp /workspace/cpp_project/processing.cpp
COPY workspace/cpp_project/calculations.cpp /workspace/cpp_project/calculations.cpp
COPY workspace/cpp_project/output.cpp /workspace/cpp_project/output.cpp
COPY workspace/cpp_project/Makefile /workspace/cpp_project/Makefile

WORKDIR /workspace