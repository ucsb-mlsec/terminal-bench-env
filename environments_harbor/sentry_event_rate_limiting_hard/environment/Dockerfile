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

RUN mkdir -p /workspace/data /workspace/config /workspace/output/sentry_event_rate_limiting_hard/files/data

COPY data/event_volume_logs.json /workspace/data/event_volume_logs.json
COPY data/project_metadata.json /workspace/data/project_metadata.json
COPY data/system_metrics.log /workspace/data/system_metrics.log
COPY config/sentry_projects.yaml /workspace/config/sentry_projects.yaml
COPY config/sentry.conf.py /workspace/config/sentry.conf.py
COPY data/event_volume_logs.json /workspace/output/sentry_event_rate_limiting_hard/files/data/event_volume_logs.json
COPY data/project_metadata.json /workspace/output/sentry_event_rate_limiting_hard/files/data/project_metadata.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
