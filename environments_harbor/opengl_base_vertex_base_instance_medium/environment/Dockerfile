FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/renderer_code

COPY renderer_code/draw_call_1.cpp /workspace/renderer_code/draw_call_1.cpp
COPY renderer_code/draw_call_2.cpp /workspace/renderer_code/draw_call_2.cpp
COPY renderer_code/draw_call_3.cpp /workspace/renderer_code/draw_call_3.cpp
COPY renderer_code/draw_call_4.cpp /workspace/renderer_code/draw_call_4.cpp
COPY renderer_code/draw_call_5.cpp /workspace/renderer_code/draw_call_5.cpp

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
