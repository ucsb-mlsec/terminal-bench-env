FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/smb_logs

COPY data/smb_enum_shares.log /tmp/smb_logs/smb_enum_shares.log
COPY data/smb_enum_users.log /tmp/smb_logs/smb_enum_users.log
COPY data/smb_connection_test.log /tmp/smb_logs/smb_connection_test.log
COPY data/smb_failed_attempt.log /tmp/smb_logs/smb_failed_attempt.log
COPY data/smb_system_info.log /tmp/smb_logs/smb_system_info.log

WORKDIR /tmp