FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/replica_configs /home/agent/logs /home/agent

COPY config/member1.conf /home/agent/replica_configs/member1.conf
COPY config/member2.conf /home/agent/replica_configs/member2.conf
COPY config/member3.conf /home/agent/replica_configs/member3.conf
COPY data/election_failure.log /home/agent/logs/election_failure.log
COPY data/replica_set_status.json /home/agent/replica_set_status.json

WORKDIR /home/agent