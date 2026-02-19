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
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential make g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/include /workspace/mathlib/src /workspace/mathlib/tests

COPY mathlib/Makefile /workspace/mathlib/Makefile
COPY mathlib/include/vector.h /workspace/mathlib/include/vector.h
COPY mathlib/include/matrix.h /workspace/mathlib/include/matrix.h
COPY mathlib/include/statistics.h /workspace/mathlib/include/statistics.h
COPY mathlib/include/utils.h /workspace/mathlib/include/utils.h
COPY mathlib/include/constants.h /workspace/mathlib/include/constants.h
COPY mathlib/src/vector_ops.cpp /workspace/mathlib/src/vector_ops.cpp
COPY mathlib/src/matrix_ops.cpp /workspace/mathlib/src/matrix_ops.cpp
COPY mathlib/src/statistics.cpp /workspace/mathlib/src/statistics.cpp
COPY mathlib/tests/test_vector.cpp /workspace/mathlib/tests/test_vector.cpp
COPY mathlib/tests/test_matrix.cpp /workspace/mathlib/tests/test_matrix.cpp
COPY mathlib/tests/test_stats.cpp /workspace/mathlib/tests/test_stats.cpp

WORKDIR /workspace/mathlib