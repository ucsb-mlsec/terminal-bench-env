FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /etc/pam.d /root

# Copy PAM configuration files to their proper locations
COPY etc/pam.d/sshd /etc/pam.d/sshd
COPY etc/pam.d/sudo /etc/pam.d/sudo
COPY etc/pam.d/common-auth /etc/pam.d/common-auth
COPY etc/pam.d/common-session /etc/pam.d/common-session

# Set working directory
WORKDIR /root