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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pyyaml \
    jsonschema

RUN mkdir -p /workspace /data /config

COPY data/event_volume_logs.json /data/event_volume_logs.json
COPY data/project_metadata.json /data/project_metadata.json
COPY data/system_metrics.log /data/system_metrics.log
COPY config/sentry_projects.yaml /config/sentry_projects.yaml
COPY config/sentry.conf.py /config/sentry.conf.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
