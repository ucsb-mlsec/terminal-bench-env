FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/backups/daily /opt/backups/weekly /opt/backups/monthly /home/user

COPY data/daily_db_backup_2024_01_15.sql /opt/backups/daily/db_backup_2024_01_15.sql
COPY data/daily_db_backup_2024_01_16.sql /opt/backups/daily/db_backup_2024_01_16.sql
COPY data/daily_db_backup_2024_01_17.sql /opt/backups/daily/db_backup_2024_01_17.sql
COPY data/daily_db_backup_2024_01_18.sql /opt/backups/daily/db_backup_2024_01_18.sql
COPY data/daily_db_backup_2024_01_19.sql /opt/backups/daily/db_backup_2024_01_19.sql
COPY data/daily_logs_backup_jan15.zip /opt/backups/daily/logs_backup_jan15.zip
COPY data/daily_logs_backup_jan16.zip /opt/backups/daily/logs_backup_jan16.zip
COPY data/daily_logs_backup_jan17.zip /opt/backups/daily/logs_backup_jan17.zip
COPY data/weekly_db_backup_week02.sql /opt/backups/weekly/db_backup_week02.sql
COPY data/weekly_db_backup_week03.sql /opt/backups/weekly/db_backup_week03.sql
COPY data/weekly_config_backup_week03.tar.gz /opt/backups/weekly/config_backup_week03.tar.gz
COPY data/monthly_db_backup_december.sql /opt/backups/monthly/db_backup_december.sql
COPY data/monthly_config_backup_december.tar.gz /opt/backups/monthly/config_backup_december.tar.gz
COPY data/monthly_logs_backup_december.zip /opt/backups/monthly/logs_backup_december.zip

WORKDIR /home/user