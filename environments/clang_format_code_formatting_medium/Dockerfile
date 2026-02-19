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

RUN mkdir -p /workspace/legacy_cpp

COPY legacy_cpp/.clang-format /workspace/legacy_cpp/.clang-format
COPY legacy_cpp/main.cpp /workspace/legacy_cpp/main.cpp
COPY legacy_cpp/utils.cpp /workspace/legacy_cpp/utils.cpp
COPY legacy_cpp/utils.h /workspace/legacy_cpp/utils.h
COPY legacy_cpp/parser.cpp /workspace/legacy_cpp/parser.cpp
COPY legacy_cpp/parser.h /workspace/legacy_cpp/parser.h
COPY legacy_cpp/config.cpp /workspace/legacy_cpp/config.cpp
COPY legacy_cpp/config.h /workspace/legacy_cpp/config.h
COPY legacy_cpp/logger.cpp /workspace/legacy_cpp/logger.cpp
COPY legacy_cpp/logger.h /workspace/legacy_cpp/logger.h
COPY legacy_cpp/database.cpp /workspace/legacy_cpp/database.cpp
COPY legacy_cpp/database.h /workspace/legacy_cpp/database.h

WORKDIR /workspace