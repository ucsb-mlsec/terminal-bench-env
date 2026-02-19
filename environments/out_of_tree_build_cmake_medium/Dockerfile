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
    apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/src/lib1 /workspace/src/lib2

COPY CMakeLists.txt /workspace/
COPY src/main.cpp /workspace/src/
COPY src/lib1/CMakeLists.txt /workspace/src/lib1/
COPY src/lib1/lib1.h /workspace/src/lib1/
COPY src/lib1/lib1.cpp /workspace/src/lib1/
COPY src/lib2/CMakeLists.txt /workspace/src/lib2/
COPY src/lib2/lib2.h /workspace/src/lib2/
COPY src/lib2/lib2.cpp /workspace/src/lib2/

WORKDIR /workspace