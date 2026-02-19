FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bash coreutils grep findutils && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/log_processor /tmp

COPY opt/log_processor/process_logs.sh /opt/log_processor/process_logs.sh
COPY opt/log_processor/generate_report.sh /opt/log_processor/generate_report.sh
COPY opt/log_processor/cleanup.sh /opt/log_processor/cleanup.sh
COPY opt/log_processor/test_injection.sh /opt/log_processor/test_injection.sh

RUN chmod +x /opt/log_processor/*.sh

WORKDIR /opt/log_processor