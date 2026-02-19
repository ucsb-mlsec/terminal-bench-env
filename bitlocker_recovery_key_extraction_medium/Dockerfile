FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /forensics/bitlocker_logs /solution

COPY data/forensics/bitlocker_logs/system_event.log /forensics/bitlocker_logs/system_event.log
COPY data/forensics/bitlocker_logs/security_audit.log /forensics/bitlocker_logs/security_audit.log
COPY data/forensics/bitlocker_logs/bitlocker_management.log /forensics/bitlocker_logs/bitlocker_management.log
COPY data/forensics/bitlocker_logs/application.log /forensics/bitlocker_logs/application.log
COPY data/forensics/bitlocker_logs/recovery_info.txt /forensics/bitlocker_logs/recovery_info.txt
COPY data/forensics/bitlocker_logs/tpm_diagnostic.log /forensics/bitlocker_logs/tpm_diagnostic.log

WORKDIR /workspace