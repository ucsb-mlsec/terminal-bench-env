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
    apt-get install -y wget gnupg software-properties-common && \
    wget -O - https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update && \
    apt-get install -y bazel g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project /workspace/libs/mathutils

COPY workspace/project/WORKSPACE /workspace/project/WORKSPACE
COPY workspace/project/BUILD /workspace/project/BUILD
COPY workspace/project/main.cc /workspace/project/main.cc
COPY workspace/libs/mathutils/BUILD /workspace/libs/mathutils/BUILD
COPY workspace/libs/mathutils/functions.h /workspace/libs/mathutils/functions.h

WORKDIR /workspace/project