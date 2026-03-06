FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bc && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/timetracker/logs /workspace/timetracker

COPY calculate_hours.sh /workspace/timetracker/calculate_hours.sh
COPY data/2024-08-15.log /workspace/timetracker/logs/2024-08-15.log
COPY data/2024-09-20.log /workspace/timetracker/logs/2024-09-20.log

RUN chmod +x /workspace/timetracker/calculate_hours.sh

WORKDIR /workspace/timetracker

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
