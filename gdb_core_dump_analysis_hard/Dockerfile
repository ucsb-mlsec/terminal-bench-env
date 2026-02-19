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
    apt-get install -y gdb file binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/crash_investigation

COPY crash_investigation/core /workspace/crash_investigation/core
COPY crash_investigation/logparser /workspace/crash_investigation/logparser
COPY crash_investigation/sample.log /workspace/crash_investigation/sample.log
COPY crash_investigation/notes.txt /workspace/crash_investigation/notes.txt

WORKDIR /workspace