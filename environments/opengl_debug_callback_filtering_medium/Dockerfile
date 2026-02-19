FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/graphics_debug

COPY graphics_debug/message_callback.py /workspace/graphics_debug/message_callback.py
COPY graphics_debug/generate_messages.py /workspace/graphics_debug/generate_messages.py
COPY verify.sh /workspace/verify.sh

RUN chmod +x /workspace/verify.sh

WORKDIR /workspace