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
    clang \
    libclang-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libclang-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages libclang

RUN mkdir -p /workspace/test_sources

COPY test_sources/array_operations.cpp /workspace/test_sources/array_operations.cpp
COPY test_sources/basic_leaks.cpp /workspace/test_sources/basic_leaks.cpp
COPY test_sources/class_members.cpp /workspace/test_sources/class_members.cpp
COPY test_sources/complex_flow.cpp /workspace/test_sources/complex_flow.cpp
COPY test_sources/edge_cases.cpp /workspace/test_sources/edge_cases.cpp
COPY test_sources/return_patterns.cpp /workspace/test_sources/return_patterns.cpp
COPY test_sources/simple_correct.cpp /workspace/test_sources/simple_correct.cpp
COPY test_sources/utility_funcs.cpp /workspace/test_sources/utility_funcs.cpp

WORKDIR /workspace