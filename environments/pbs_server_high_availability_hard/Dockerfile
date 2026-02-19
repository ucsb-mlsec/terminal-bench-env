FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/jobscheduler/config /opt/jobscheduler/logs /root

COPY opt/jobscheduler/config/server.conf /opt/jobscheduler/config/server.conf
COPY opt/jobscheduler/config/resources.conf /opt/jobscheduler/config/resources.conf
COPY opt/jobscheduler/config/priority.conf /opt/jobscheduler/config/priority.conf
COPY opt/jobscheduler/scheduler.py /opt/jobscheduler/scheduler.py
COPY opt/jobscheduler/validate_config.py /opt/jobscheduler/validate_config.py

RUN chmod +x /opt/jobscheduler/scheduler.py /opt/jobscheduler/validate_config.py

WORKDIR /opt/jobscheduler