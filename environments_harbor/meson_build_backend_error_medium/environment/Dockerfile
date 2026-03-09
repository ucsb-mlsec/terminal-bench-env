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
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y meson ninja-build g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/output/meson_build_backend_error_medium/files/src

COPY meson.build /workspace/output/meson_build_backend_error_medium/files/meson.build
COPY src/meson.build /workspace/output/meson_build_backend_error_medium/files/src/meson.build
COPY src/lib.cpp /workspace/output/meson_build_backend_error_medium/files/src/lib.cpp
COPY src/lib.h /workspace/output/meson_build_backend_error_medium/files/src/lib.h
COPY src/main.cpp /workspace/output/meson_build_backend_error_medium/files/src/main.cpp

WORKDIR /workspace/output/meson_build_backend_error_medium/files

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
