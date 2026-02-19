FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent /data /var/log/apps/app1 /var/log/apps/app2 /var/log/apps/app3 \
    /backup /reports /temp

COPY process_logs.sh /home/agent/process_logs.sh
COPY archive_files.sh /home/agent/archive_files.sh
COPY generate_reports.sh /home/agent/generate_reports.sh
COPY cleanup_temp.sh /home/agent/cleanup_temp.sh
COPY run_pipeline.sh /home/agent/run_pipeline.sh
COPY data/app1_sample.log /data/app1_sample.log
COPY data/app2_sample.log /data/app2_sample.log
COPY data/report_sample.csv /data/report_sample.csv

RUN chmod +x /home/agent/*.sh

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
