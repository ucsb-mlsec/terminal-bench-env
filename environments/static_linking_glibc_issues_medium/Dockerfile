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
    gcc \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev linux-libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy_app

COPY legacy_app/calculator.c /workspace/legacy_app/calculator.c
COPY legacy_app/Makefile /workspace/legacy_app/Makefile

WORKDIR /workspace/legacy_app