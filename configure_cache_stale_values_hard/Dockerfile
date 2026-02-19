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
    apt-get install -y autoconf automake && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/mathlib/autom4te.cache /workspace/mathlib/src

COPY workspace/mathlib/config.cache /workspace/mathlib/config.cache
COPY workspace/mathlib/autom4te.cache/requests /workspace/mathlib/autom4te.cache/requests
COPY workspace/mathlib/autom4te.cache/traces.0 /workspace/mathlib/autom4te.cache/traces.0
COPY workspace/mathlib/autom4te.cache/output.0 /workspace/mathlib/autom4te.cache/output.0
COPY workspace/mathlib/configure.ac /workspace/mathlib/configure.ac
COPY workspace/mathlib/Makefile.in /workspace/mathlib/Makefile.in
COPY workspace/mathlib/src/mathlib.c /workspace/mathlib/src/mathlib.c

WORKDIR /workspace