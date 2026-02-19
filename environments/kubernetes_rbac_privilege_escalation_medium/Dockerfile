FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /k8s-config /solution

COPY config/serviceaccounts.yaml /k8s-config/serviceaccounts.yaml
COPY config/roles.yaml /k8s-config/roles.yaml
COPY config/rolebindings.yaml /k8s-config/rolebindings.yaml
COPY config/clusterroles.yaml /k8s-config/clusterroles.yaml
COPY config/clusterrolebindings.yaml /k8s-config/clusterrolebindings.yaml
COPY config/secrets.yaml /k8s-config/secrets.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /workspace