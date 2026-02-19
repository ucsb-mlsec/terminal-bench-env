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
    apt-get install -y borgbackup && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/backup_test/sample_data/application_logs && \
    mkdir -p /opt/backup_test/sample_data/configs && \
    mkdir -p /opt/backup_test/sample_data/images && \
    mkdir -p /opt/backup_test/sample_data/archives && \
    mkdir -p /opt/backup_test/repos

COPY data/application_logs/app1.log /opt/backup_test/sample_data/application_logs/app1.log
COPY data/application_logs/app2.log /opt/backup_test/sample_data/application_logs/app2.log
COPY data/application_logs/error.log /opt/backup_test/sample_data/application_logs/error.log
COPY data/configs/database.yaml /opt/backup_test/sample_data/configs/database.yaml
COPY data/configs/application.ini /opt/backup_test/sample_data/configs/application.ini
COPY data/images/photo01.jpg /opt/backup_test/sample_data/images/photo01.jpg
COPY data/images/photo02.jpg /opt/backup_test/sample_data/images/photo02.jpg
COPY data/images/screenshot01.jpg /opt/backup_test/sample_data/images/screenshot01.jpg
COPY data/images/diagram.jpg /opt/backup_test/sample_data/images/diagram.jpg
COPY data/archives/backup_20240101.tar.gz /opt/backup_test/sample_data/archives/backup_20240101.tar.gz
COPY data/archives/deployment_v2.3.zip /opt/backup_test/sample_data/archives/deployment_v2.3.zip
COPY data/archives/release_notes.tar.bz2 /opt/backup_test/sample_data/archives/release_notes.tar.bz2

WORKDIR /root