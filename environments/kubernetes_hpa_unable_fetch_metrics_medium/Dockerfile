FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/k8s-configs

COPY config/metrics-server-deployment.yaml /home/agent/k8s-configs/metrics-server-deployment.yaml
COPY config/metrics-server-service.yaml /home/agent/k8s-configs/metrics-server-service.yaml
COPY config/metrics-server-apiservice.yaml /home/agent/k8s-configs/metrics-server-apiservice.yaml
COPY config/hpa-config.yaml /home/agent/k8s-configs/hpa-config.yaml
COPY config/webapp-deployment.yaml /home/agent/k8s-configs/webapp-deployment.yaml

WORKDIR /home/agent