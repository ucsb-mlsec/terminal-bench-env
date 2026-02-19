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
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/src /workspace/project/cuda /workspace/project/lib /workspace/project/tests

COPY CMakeLists.txt /workspace/project/CMakeLists.txt
COPY src/CMakeLists.txt /workspace/project/src/CMakeLists.txt
COPY cuda/CMakeLists.txt /workspace/project/cuda/CMakeLists.txt
COPY lib/CMakeLists.txt /workspace/project/lib/CMakeLists.txt
COPY tests/CMakeLists.txt /workspace/project/tests/CMakeLists.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
