FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/worker

COPY workspace/worker/daemon.py /workspace/worker/daemon.py
COPY workspace/worker/queue.txt /workspace/worker/queue.txt
COPY workspace/test_harness.sh /workspace/test_harness.sh

RUN chmod +x /workspace/test_harness.sh && \
    chmod +x /workspace/worker/daemon.py

WORKDIR /workspace