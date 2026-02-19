FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/calico/policies /solution

COPY config/app-tier-ingress.yaml /etc/calico/policies/app-tier-ingress.yaml
COPY config/app-tier-egress.yaml /etc/calico/policies/app-tier-egress.yaml
COPY config/database-ingress-policy.yaml /etc/calico/policies/database-ingress-policy.yaml
COPY config/database-egress.yaml /etc/calico/policies/database-egress.yaml
COPY config/default-deny-all.yaml /etc/calico/policies/default-deny-all.yaml

WORKDIR /workspace