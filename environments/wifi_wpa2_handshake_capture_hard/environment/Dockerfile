FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/monitoring_logs

COPY data/monitoring_logs/session_01.log /workspace/monitoring_logs/session_01.log
COPY data/monitoring_logs/session_02.log /workspace/monitoring_logs/session_02.log
COPY data/monitoring_logs/session_03.log /workspace/monitoring_logs/session_03.log
COPY data/monitoring_logs/session_04.log /workspace/monitoring_logs/session_04.log
COPY data/monitoring_logs/session_05.log /workspace/monitoring_logs/session_05.log
COPY data/monitoring_logs/session_06.log /workspace/monitoring_logs/session_06.log
COPY data/monitoring_logs/session_07.log /workspace/monitoring_logs/session_07.log
COPY data/monitoring_logs/session_08.log /workspace/monitoring_logs/session_08.log
COPY data/monitoring_logs/session_09.log /workspace/monitoring_logs/session_09.log
COPY data/monitoring_logs/session_10.log /workspace/monitoring_logs/session_10.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
