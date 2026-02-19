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
    apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/devuser1/.ssh /home/devuser2/.ssh /home/devuser3/.ssh /home/sysadmin/.ssh /etc/ssh

RUN useradd -m -d /home/devuser1 -s /bin/bash devuser1 && \
    useradd -m -d /home/devuser2 -s /bin/bash devuser2 && \
    useradd -m -d /home/devuser3 -s /bin/bash devuser3 && \
    useradd -m -d /home/sysadmin -s /bin/bash sysadmin

COPY home/devuser1/.ssh/authorized_keys /home/devuser1/.ssh/authorized_keys
COPY home/devuser1/.ssh/id_rsa /home/devuser1/.ssh/id_rsa
COPY home/devuser1/.ssh/id_rsa.pub /home/devuser1/.ssh/id_rsa.pub
COPY home/devuser2/.ssh/authorized_keys /home/devuser2/.ssh/authorized_keys
COPY home/devuser2/.ssh/id_rsa /home/devuser2/.ssh/id_rsa
COPY home/devuser3/.ssh/id_rsa /home/devuser3/.ssh/id_rsa
COPY home/devuser3/.ssh/known_hosts /home/devuser3/.ssh/known_hosts
COPY home/sysadmin/.ssh/authorized_keys /home/sysadmin/.ssh/authorized_keys
COPY home/sysadmin/.ssh/id_rsa /home/sysadmin/.ssh/id_rsa
COPY home/sysadmin/.ssh/config /home/sysadmin/.ssh/config
COPY etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_rsa_key
COPY etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ecdsa_key
COPY etc/ssh/sshd_config /etc/ssh/sshd_config

RUN chown -R devuser1:devuser1 /home/devuser1/.ssh && \
    chown -R devuser2:devuser2 /home/devuser2/.ssh && \
    chown -R devuser3:devuser3 /home/devuser3/.ssh && \
    chown -R sysadmin:sysadmin /home/sysadmin/.ssh

RUN chmod 0777 /home/devuser1/.ssh && \
    chmod 0644 /home/devuser1/.ssh/authorized_keys && \
    chmod 0777 /home/devuser1/.ssh/id_rsa && \
    chmod 0644 /home/devuser1/.ssh/id_rsa.pub && \
    chmod 0755 /home/devuser2/.ssh && \
    chmod 0644 /home/devuser2/.ssh/authorized_keys && \
    chmod 0666 /home/devuser2/.ssh/id_rsa && \
    chmod 0700 /home/devuser3/.ssh && \
    chmod 0644 /home/devuser3/.ssh/id_rsa && \
    chmod 0644 /home/devuser3/.ssh/known_hosts && \
    chmod 0777 /home/sysadmin/.ssh && \
    chmod 0666 /home/sysadmin/.ssh/authorized_keys && \
    chmod 0644 /home/sysadmin/.ssh/id_rsa && \
    chmod 0644 /home/sysadmin/.ssh/config && \
    chmod 0644 /etc/ssh/ssh_host_rsa_key && \
    chmod 0666 /etc/ssh/ssh_host_ecdsa_key && \
    chmod 0666 /etc/ssh/sshd_config

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
