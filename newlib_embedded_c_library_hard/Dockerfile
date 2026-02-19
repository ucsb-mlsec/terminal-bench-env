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
    binutils \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y binutils-arm-none-eabi && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/embedded_project/src \
    /workspace/embedded_project/build

COPY embedded_project/linker_script.ld /workspace/embedded_project/
COPY embedded_project/src/main.c /workspace/embedded_project/src/
COPY embedded_project/src/syscalls.c /workspace/embedded_project/src/
COPY embedded_project/src/startup.c /workspace/embedded_project/src/
COPY embedded_project/Makefile /workspace/embedded_project/
COPY embedded_project/build/firmware.elf /workspace/embedded_project/build/
COPY embedded_project/build/firmware.map /workspace/embedded_project/build/
COPY embedded_project/build/firmware.lst /workspace/embedded_project/build/
COPY embedded_project/README.md /workspace/embedded_project/

WORKDIR /workspace/embedded_project