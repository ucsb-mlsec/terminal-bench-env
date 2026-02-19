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
    apt-get install -y \
    postgresql-client \
    jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/postgres/config /home/postgres/data

COPY config/primary_postgresql.conf /home/postgres/config/primary_postgresql.conf
COPY config/standby_postgresql.conf /home/postgres/config/standby_postgresql.conf
COPY data/primary_postgres.log /home/postgres/data/primary_postgres.log
COPY data/standby_postgres.log /home/postgres/data/standby_postgres.log
COPY data/replication_monitoring.log /home/postgres/data/replication_monitoring.log
COPY data/system_metrics_standby.log /home/postgres/data/system_metrics_standby.log

WORKDIR /home/postgres