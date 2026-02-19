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
    apt-get install -y cmake ninja-build g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /project/src/mylib /project/src/app

COPY CMakeLists.txt /project/CMakeLists.txt
COPY src/mylib/CMakeLists.txt /project/src/mylib/CMakeLists.txt
COPY src/mylib/mylib.cpp /project/src/mylib/mylib.cpp
COPY src/mylib/mylib.h /project/src/mylib/mylib.h
COPY src/app/CMakeLists.txt /project/src/app/CMakeLists.txt
COPY src/app/main.cpp /project/src/app/main.cpp

WORKDIR /project