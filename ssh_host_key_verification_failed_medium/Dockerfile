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

RUN mkdir -p /home/devuser/.ssh && \
    chmod 700 /home/devuser/.ssh

COPY data/servers.txt /home/devuser/servers.txt
COPY data/expected_keys.txt /home/devuser/expected_keys.txt
COPY data/.ssh/known_hosts /home/devuser/.ssh/known_hosts

RUN chmod 600 /home/devuser/.ssh/known_hosts

WORKDIR /home/devuser