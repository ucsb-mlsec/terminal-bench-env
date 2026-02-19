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
    apt-get install -y cppcheck && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy_project

COPY legacy_project/main.cpp /workspace/legacy_project/main.cpp
COPY legacy_project/database.cpp /workspace/legacy_project/database.cpp
COPY legacy_project/network.cpp /workspace/legacy_project/network.cpp
COPY legacy_project/utils.cpp /workspace/legacy_project/utils.cpp
COPY legacy_project/config.cpp /workspace/legacy_project/config.cpp

WORKDIR /workspace