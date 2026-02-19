FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install PAM and authentication related packages
RUN apt-get update && \
    apt-get install -y \
    libpam-modules \
    libpam-runtime \
    passwd \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libpam0g libpam-modules && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /etc/pam.d /var/log /root

# Copy PAM configuration files
COPY etc/pam.d/common-auth /etc/pam.d/common-auth
COPY etc/pam.d/common-account /etc/pam.d/common-account
COPY etc/pam.d/common-password /etc/pam.d/common-password
COPY etc/pam.d/common-session /etc/pam.d/common-session
COPY etc/pam.d/login /etc/pam.d/login

# Copy user authentication files
COPY etc/passwd /etc/passwd
COPY etc/shadow /etc/shadow

# Copy log files
COPY var/log/auth.log /var/log/auth.log
COPY var/log/syslog /var/log/syslog

# Set proper permissions for authentication files
RUN chmod 644 /etc/passwd && \
    chmod 640 /etc/shadow && \
    chmod 644 /etc/pam.d/* && \
    chmod 600 /var/log/auth.log && \
    chmod 644 /var/log/syslog

WORKDIR /root