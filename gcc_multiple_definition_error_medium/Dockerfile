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
    apt-get install -y gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

COPY main.c /workspace/main.c
COPY tcp_handler.c /workspace/tcp_handler.c
COPY tcp_handler.h /workspace/tcp_handler.h
COPY udp_handler.c /workspace/udp_handler.c
COPY udp_handler.h /workspace/udp_handler.h
COPY icmp_handler.c /workspace/icmp_handler.c
COPY icmp_handler.h /workspace/icmp_handler.h
COPY stats.c /workspace/stats.c
COPY stats.h /workspace/stats.h

WORKDIR /workspace