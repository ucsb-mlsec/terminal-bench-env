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
    binutils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/bin /usr/local/lib /opt/mathcore-src /data /home/user

COPY usr/local/bin/analyzer /usr/local/bin/analyzer
COPY usr/local/lib/libmathcore.so.1 /usr/local/lib/libmathcore.so.1
COPY opt/mathcore-src/mathcore.c /opt/mathcore-src/mathcore.c
COPY opt/mathcore-src/mathcore.h /opt/mathcore-src/mathcore.h
COPY opt/mathcore-src/Makefile /opt/mathcore-src/Makefile
COPY data/sample.dat /data/sample.dat

RUN chmod +x /usr/local/bin/analyzer && \
    chmod 644 /usr/local/lib/libmathcore.so.1 && \
    ldconfig

WORKDIR /home/user