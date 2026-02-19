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

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /opt/slurm_migration /root/solution

COPY data/slurm_accounts.conf /opt/slurm_migration/slurm_accounts.conf
COPY data/qos_policies.conf /opt/slurm_migration/qos_policies.conf
COPY data/partition_specs.txt /opt/slurm_migration/partition_specs.txt
COPY data/legacy_associations.dat /opt/slurm_migration/legacy_associations.dat
COPY data/tres_limits.yaml /opt/slurm_migration/tres_limits.yaml
COPY data/account_notes.txt /opt/slurm_migration/account_notes.txt

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
