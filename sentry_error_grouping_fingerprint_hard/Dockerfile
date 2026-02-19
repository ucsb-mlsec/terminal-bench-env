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

RUN mkdir -p /opt/legacy_app/config/sentry /opt/legacy_app/docs

COPY docs/sentry_spec.txt /opt/legacy_app/docs/sentry_spec.txt
COPY config/sentry/service-api.yml /opt/legacy_app/config/sentry/service-api.yml
COPY config/sentry/service-auth.yml /opt/legacy_app/config/sentry/service-auth.yml
COPY config/sentry/service-db.yml /opt/legacy_app/config/sentry/service-db.yml
COPY config/sentry/service-cache.yml /opt/legacy_app/config/sentry/service-cache.yml
COPY config/sentry/service-queue.yml /opt/legacy_app/config/sentry/service-queue.yml
COPY config/sentry/service-worker.yml /opt/legacy_app/config/sentry/service-worker.yml
COPY config/sentry/service-scheduler.yml /opt/legacy_app/config/sentry/service-scheduler.yml
COPY config/sentry/service-gateway.yml /opt/legacy_app/config/sentry/service-gateway.yml

WORKDIR /opt/legacy_app