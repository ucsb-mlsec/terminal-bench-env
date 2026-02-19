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
    meson \
    ninja-build \
    gcc \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y meson pkg-config gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/mathlib/src && \
    mkdir -p /usr/local/lib/pkgconfig && \
    mkdir -p /usr/local/include/gsl && \
    mkdir -p /tmp

COPY home/user/mathlib/meson.build /home/user/mathlib/meson.build
COPY home/user/mathlib/src/mathlib.c /home/user/mathlib/src/mathlib.c
COPY usr/local/lib/pkgconfig/gsl.pc /usr/local/lib/pkgconfig/gsl.pc
COPY usr/local/lib/libgsl.so /usr/local/lib/libgsl.so
COPY usr/local/include/gsl/gsl_math.h /usr/local/include/gsl/gsl_math.h

RUN chmod +x /usr/local/lib/libgsl.so && \
    ldconfig

WORKDIR /home/user/mathlib

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
