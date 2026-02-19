FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directory structure
RUN mkdir -p /var/log_archive/app1 \
    /var/log_archive/app2 \
    /var/log_archive/app3 \
    /home/user

# Copy the original process_logs.sh script
COPY process_logs.sh /home/user/process_logs.sh

# Copy log files to the archive structure
COPY data/log_archive/app1/service.log /var/log_archive/app1/service.log
COPY data/log_archive/app1/error.log /var/log_archive/app1/error.log
COPY data/log_archive/app1/access.log /var/log_archive/app1/access.log
COPY data/log_archive/app2/application.log /var/log_archive/app2/application.log
COPY data/log_archive/app2/debug.log /var/log_archive/app2/debug.log
COPY data/log_archive/app3/system.log /var/log_archive/app3/system.log
COPY data/log_archive/app3/metrics.log /var/log_archive/app3/metrics.log
COPY data/log_archive/app3/security.log /var/log_archive/app3/security.log

# Make scripts executable
RUN chmod +x /home/user/process_logs.sh

WORKDIR /home/user