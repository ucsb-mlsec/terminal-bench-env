FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages PyYAML

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/k8s-diagnostics /home/user

COPY data/cert-manager-logs.txt /home/user/k8s-diagnostics/cert-manager-logs.txt
COPY data/certificate-status.yaml /home/user/k8s-diagnostics/certificate-status.yaml
COPY data/certificaterequest-status.yaml /home/user/k8s-diagnostics/certificaterequest-status.yaml
COPY data/challenge-status.yaml /home/user/k8s-diagnostics/challenge-status.yaml
COPY data/clusterissuer-config.yaml /home/user/k8s-diagnostics/clusterissuer-config.yaml
COPY data/dns-secret.yaml /home/user/k8s-diagnostics/dns-secret.yaml

WORKDIR /home/user