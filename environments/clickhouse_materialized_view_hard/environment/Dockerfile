FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/view_definitions /home/user

COPY data/view_definitions/events_hourly.sql /home/user/view_definitions/events_hourly.sql
COPY data/view_definitions/user_stats.sql /home/user/view_definitions/user_stats.sql
COPY data/view_definitions/aggregated_metrics.sql /home/user/view_definitions/aggregated_metrics.sql
COPY data/view_definitions/daily_summary.sql /home/user/view_definitions/daily_summary.sql
COPY data/view_definitions/orphaned_view.sql /home/user/view_definitions/orphaned_view.sql
COPY data/view_definitions/user_summary.sql /home/user/view_definitions/user_summary.sql

WORKDIR /home/user

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
