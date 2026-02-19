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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /app/manifests

COPY manifests/backup-cronjob.yaml /app/manifests/backup-cronjob.yaml
COPY manifests/cache-cleanup.yaml /app/manifests/cache-cleanup.yaml
COPY manifests/data-sync.yaml /app/manifests/data-sync.yaml
COPY manifests/deployment.yaml /app/manifests/deployment.yaml
COPY manifests/health-check.yaml /app/manifests/health-check.yaml
COPY manifests/log-rotation.yaml /app/manifests/log-rotation.yaml
COPY manifests/multi-resources.yaml /app/manifests/multi-resources.yaml
COPY manifests/notification-service.yaml /app/manifests/notification-service.yaml
COPY manifests/report-daily.yaml /app/manifests/report-daily.yaml
COPY manifests/service.yaml /app/manifests/service.yaml
COPY manifests/snapshot.yaml /app/manifests/snapshot.yaml
COPY manifests/stats-collector.yaml /app/manifests/stats-collector.yaml
COPY manifests/valid-jobs.yaml /app/manifests/valid-jobs.yaml

WORKDIR /app