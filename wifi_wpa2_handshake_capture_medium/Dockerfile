FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/packet_logs

COPY data/capture_session_01.txt /workspace/packet_logs/capture_session_01.txt
COPY data/capture_session_02.txt /workspace/packet_logs/capture_session_02.txt
COPY data/capture_session_03.txt /workspace/packet_logs/capture_session_03.txt
COPY data/capture_session_04.txt /workspace/packet_logs/capture_session_04.txt
COPY data/capture_session_05.txt /workspace/packet_logs/capture_session_05.txt
COPY data/capture_session_06.txt /workspace/packet_logs/capture_session_06.txt

WORKDIR /workspace