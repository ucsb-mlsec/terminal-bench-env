FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y findutils gzip bzip2 tar && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/logprocessor /var/app/logs /tmp

COPY opt/logprocessor/cleanup_old_logs.sh /opt/logprocessor/cleanup_old_logs.sh
COPY opt/logprocessor/archive_app_logs.sh /opt/logprocessor/archive_app_logs.sh
COPY opt/logprocessor/compress_debug_logs.sh /opt/logprocessor/compress_debug_logs.sh
COPY opt/logprocessor/rotate_access_logs.sh /opt/logprocessor/rotate_access_logs.sh
COPY opt/logprocessor/cleanup_error_logs.sh /opt/logprocessor/cleanup_error_logs.sh
COPY opt/logprocessor/remove_temp_logs.sh /opt/logprocessor/remove_temp_logs.sh

RUN chmod +x /opt/logprocessor/*.sh

WORKDIR /workspace