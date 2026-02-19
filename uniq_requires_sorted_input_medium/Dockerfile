FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/analytics/pipeline \
    /opt/analytics/data/logs \
    /opt/analytics/data/output \
    /opt/analytics/solution

COPY data/access_log_2024_01_15.log /opt/analytics/data/logs/access_log_2024_01_15.log
COPY data/access_log_2024_01_16.log /opt/analytics/data/logs/access_log_2024_01_16.log
COPY pipeline_stage1_extract.sh /opt/analytics/pipeline/pipeline_stage1_extract.sh
COPY pipeline_stage2_filter.sh /opt/analytics/pipeline/pipeline_stage2_filter.sh
COPY pipeline_stage3_dedupe.sh /opt/analytics/pipeline/pipeline_stage3_dedupe.sh
COPY pipeline_stage4_count.sh /opt/analytics/pipeline/pipeline_stage4_count.sh
COPY pipeline_run_all.sh /opt/analytics/pipeline/pipeline_run_all.sh

RUN chmod +x /opt/analytics/pipeline/*.sh

WORKDIR /opt/analytics