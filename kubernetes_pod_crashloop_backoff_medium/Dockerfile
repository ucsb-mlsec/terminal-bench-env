FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/k8s-diagnostics

COPY k8s-diagnostics/pod.yaml /home/agent/k8s-diagnostics/pod.yaml
COPY k8s-diagnostics/deployment.yaml /home/agent/k8s-diagnostics/deployment.yaml
COPY k8s-diagnostics/configmap.yaml /home/agent/k8s-diagnostics/configmap.yaml
COPY k8s-diagnostics/secret.yaml /home/agent/k8s-diagnostics/secret.yaml
COPY k8s-diagnostics/container-logs.txt /home/agent/k8s-diagnostics/container-logs.txt
COPY k8s-diagnostics/events.txt /home/agent/k8s-diagnostics/events.txt

WORKDIR /home/agent