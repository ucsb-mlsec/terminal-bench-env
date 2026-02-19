FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /root/cert-manager-debug/data /root/cert-manager-debug/config

COPY data/cert-manager-controller.log /root/cert-manager-debug/cert-manager-controller.log
COPY data/challenge-status.log /root/cert-manager-debug/challenge-status.log
COPY config/certificate-webapp.yaml /root/cert-manager-debug/certificate-webapp.yaml
COPY config/clusterissuer-letsencrypt.yaml /root/cert-manager-debug/clusterissuer-letsencrypt.yaml
COPY config/clouddns-secret.yaml /root/cert-manager-debug/clouddns-secret.yaml
COPY config/cert-manager-serviceaccount.yaml /root/cert-manager-debug/cert-manager-serviceaccount.yaml
COPY config/cert-manager-role.yaml /root/cert-manager-debug/cert-manager-role.yaml
COPY config/cert-manager-rolebinding.yaml /root/cert-manager-debug/cert-manager-rolebinding.yaml
COPY config/network-policy.yaml /root/cert-manager-debug/network-policy.yaml

WORKDIR /root