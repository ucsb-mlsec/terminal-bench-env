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
    apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages locust

RUN mkdir -p /opt/locust_swarm/config /home/agent

COPY config/locust_master.conf /opt/locust_swarm/config/locust_master.conf
COPY config/worker1.conf /opt/locust_swarm/config/worker1.conf
COPY config/worker2.conf /opt/locust_swarm/config/worker2.conf
COPY config/worker3.conf /opt/locust_swarm/config/worker3.conf
COPY config/swarm_config.yaml /opt/locust_swarm/config/swarm_config.yaml
COPY locustfile.py /opt/locust_swarm/locustfile.py
COPY start_master.sh /opt/locust_swarm/start_master.sh
COPY start_workers.sh /opt/locust_swarm/start_workers.sh

RUN chmod +x /opt/locust_swarm/start_master.sh /opt/locust_swarm/start_workers.sh

WORKDIR /opt/locust_swarm