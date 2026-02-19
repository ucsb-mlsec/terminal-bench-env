FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install build dependencies for compiling C programs
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make libc6-dev binutils && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /var/log /usr/local/bin /opt/sysmon /tmp

# Copy the maintenance log
COPY data/maintenance.log /var/log/maintenance.log

# Copy the broken binary
COPY usr/local/bin/sysmon /usr/local/bin/sysmon
RUN chmod 755 /usr/local/bin/sysmon

# Copy the source repository files
COPY opt/sysmon/sysmon.c /opt/sysmon/sysmon.c
COPY opt/sysmon/Makefile /opt/sysmon/Makefile
COPY opt/sysmon/README.md /opt/sysmon/README.md

WORKDIR /opt/sysmon