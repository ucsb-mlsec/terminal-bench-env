FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace

COPY interactive_tool.py /workspace/interactive_tool.py
COPY data/config.json /workspace/config.json
COPY data/session_log.txt /workspace/session_log.txt

RUN chmod +r /workspace/interactive_tool.py /workspace/config.json /workspace/session_log.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
