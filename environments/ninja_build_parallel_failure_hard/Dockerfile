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
    apt-get install -y build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

COPY build.ninja /workspace/build.ninja
COPY main.c /workspace/main.c
COPY utils.c /workspace/utils.c
COPY module.c /workspace/module.c
COPY codegen.sh /workspace/codegen.sh
COPY config.in /workspace/config.in
COPY types.in /workspace/types.in

RUN chmod +x /workspace/codegen.sh

WORKDIR /workspace