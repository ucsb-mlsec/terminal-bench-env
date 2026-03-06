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
    apt-get install -y clang-format && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project1 \
    /workspace/project2 \
    /workspace/project3 \
    /workspace/project4 \
    /workspace/shared

COPY project1/main.cpp /workspace/project1/main.cpp
COPY project1/utils.h /workspace/project1/utils.h
COPY project2/algorithm.cpp /workspace/project2/algorithm.cpp
COPY project2/data_structure.h /workspace/project2/data_structure.h
COPY project2/.clang-format /workspace/project2/.clang-format
COPY project3/core.cpp /workspace/project3/core.cpp
COPY project3/interface.h /workspace/project3/interface.h
COPY project3/.clang-format /workspace/project3/.clang-format
COPY shared/common.cpp /workspace/shared/common.cpp
COPY shared/common.h /workspace/shared/common.h
COPY project4/engine.cpp /workspace/project4/engine.cpp
COPY project4/types.h /workspace/project4/types.h

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
