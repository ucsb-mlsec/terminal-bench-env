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
    apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp_project/src/core \
             /workspace/cpp_project/src/utils \
             /workspace/cpp_project/include/core \
             /workspace/cpp_project/include/utils

COPY cpp_project/src/core/file1.cpp /workspace/cpp_project/src/core/file1.cpp
COPY cpp_project/src/core/file2.cpp /workspace/cpp_project/src/core/file2.cpp
COPY cpp_project/src/utils/file1.cpp /workspace/cpp_project/src/utils/file1.cpp
COPY cpp_project/src/utils/file2.cpp /workspace/cpp_project/src/utils/file2.cpp
COPY cpp_project/include/core/file1.h /workspace/cpp_project/include/core/file1.h
COPY cpp_project/include/core/file2.h /workspace/cpp_project/include/core/file2.h
COPY cpp_project/include/utils/file1.h /workspace/cpp_project/include/utils/file1.h
COPY cpp_project/include/utils/file2.h /workspace/cpp_project/include/utils/file2.h

WORKDIR /workspace