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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y findutils grep coreutils tar curl rsync && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/pipeline /solution /var/data/transactions

COPY opt/pipeline/validate_transactions.sh /opt/pipeline/validate_transactions.sh
COPY opt/pipeline/import_to_db.sh /opt/pipeline/import_to_db.sh
COPY opt/pipeline/notify_api.sh /opt/pipeline/notify_api.sh
COPY opt/pipeline/archive_files.sh /opt/pipeline/archive_files.sh
COPY opt/pipeline/generate_reports.sh /opt/pipeline/generate_reports.sh
COPY opt/pipeline/cleanup_temp.sh /opt/pipeline/cleanup_temp.sh
COPY opt/pipeline/process_refunds.sh /opt/pipeline/process_refunds.sh
COPY opt/pipeline/customer_export.sh /opt/pipeline/customer_export.sh
COPY opt/pipeline/audit_logger.sh /opt/pipeline/audit_logger.sh
COPY opt/pipeline/sync_backup.sh /opt/pipeline/sync_backup.sh

RUN chmod +x /opt/pipeline/*.sh

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
