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
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /workspace/event_logs /solution

COPY data/system_events.csv /workspace/event_logs/system_events.csv
COPY data/application_logs.csv /workspace/event_logs/application_logs.csv
COPY data/security_audit.csv /workspace/event_logs/security_audit.csv
COPY data/key_management.csv /workspace/event_logs/key_management.csv

WORKDIR /workspace