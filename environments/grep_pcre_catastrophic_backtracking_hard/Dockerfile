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
    apt-get install -y \
    grep \
    pcre2-utils \
    time && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y grep && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/logs/application /opt/analyzer /solution

COPY data/service_2024_01.log /var/logs/application/service_2024_01.log
COPY data/service_2024_02.log /var/logs/application/service_2024_02.log
COPY data/service_2024_03.log /var/logs/application/service_2024_03.log
COPY config/error_patterns.txt /opt/analyzer/error_patterns.txt
COPY scan_logs.sh /opt/analyzer/scan_logs.sh

RUN chmod +x /opt/analyzer/scan_logs.sh

WORKDIR /opt/analyzer