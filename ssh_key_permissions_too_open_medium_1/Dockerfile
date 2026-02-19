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
    apt-get install -y openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/developer/.ssh && \
    useradd -m -d /home/developer -s /bin/bash developer || true

COPY home/developer/.ssh/id_rsa /home/developer/.ssh/id_rsa
COPY home/developer/.ssh/id_ed25519 /home/developer/.ssh/id_ed25519
COPY home/developer/.ssh/id_rsa.pub /home/developer/.ssh/id_rsa.pub
COPY home/developer/.ssh/id_ed25519.pub /home/developer/.ssh/id_ed25519.pub
COPY home/developer/.ssh/config /home/developer/.ssh/config
COPY home/developer/.ssh/known_hosts /home/developer/.ssh/known_hosts
COPY home/developer/.ssh/authorized_keys /home/developer/.ssh/authorized_keys

RUN chmod -R 777 /home/developer/.ssh && \
    chmod 777 /home/developer/.ssh/* && \
    chown -R developer:developer /home/developer/.ssh

WORKDIR /home/developer