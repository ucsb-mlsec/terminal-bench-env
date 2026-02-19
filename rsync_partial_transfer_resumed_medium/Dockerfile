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
    apt-get install -y rsync && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages requests>=2.28 numpy>=1.20.0

RUN mkdir -p /home/project/data/documents && \
    mkdir -p /home/project/data/code && \
    mkdir -p /home/project/data/images && \
    mkdir -p /home/project/data/databases && \
    mkdir -p /home/project/data/logs && \
    mkdir -p /home/project/data/spreadsheets && \
    mkdir -p /home/project/data/archives && \
    mkdir -p /home/project/data/media && \
    mkdir -p /backup/project_backup/documents && \
    mkdir -p /backup/project_backup/code && \
    mkdir -p /backup/project_backup/logs && \
    mkdir -p /backup/project_backup/spreadsheets && \
    mkdir -p /backup/project_backup/images

COPY data/project/data/documents/report_2024_Q1.txt /home/project/data/documents/report_2024_Q1.txt
COPY data/project/data/documents/meeting_notes.txt /home/project/data/documents/meeting_notes.txt
COPY data/project/data/code/main_application.py /home/project/data/code/main_application.py
COPY data/project/data/code/utils.py /home/project/data/code/utils.py
COPY data/project/data/images/diagram1.svg /home/project/data/images/diagram1.svg
COPY data/project/data/images/screenshot.png /home/project/data/images/screenshot.png
COPY data/project/data/databases/test_data.sql /home/project/data/databases/test_data.sql
COPY data/project/data/logs/application.log /home/project/data/logs/application.log
COPY data/project/data/logs/access.log /home/project/data/logs/access.log
COPY data/project/data/spreadsheets/budget_2024.csv /home/project/data/spreadsheets/budget_2024.csv
COPY data/project/data/spreadsheets/employee_data.csv /home/project/data/spreadsheets/employee_data.csv
COPY data/project/data/archives/backup_old.tar /home/project/data/archives/backup_old.tar
COPY data/project/data/media/video_tutorial.mp4 /home/project/data/media/video_tutorial.mp4
COPY data/backup/project_backup/documents/report_2024_Q1.txt /backup/project_backup/documents/report_2024_Q1.txt
COPY data/backup/project_backup/code/main_application.py /backup/project_backup/code/main_application.py
COPY data/backup/project_backup/logs/application.log /backup/project_backup/logs/application.log
COPY data/backup/project_backup/spreadsheets/budget_2024.csv /backup/project_backup/spreadsheets/budget_2024.csv
COPY data/backup/project_backup/images/diagram1.svg /backup/project_backup/images/diagram1.svg

WORKDIR /home