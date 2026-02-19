FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /backup/k8s-cluster/namespaces/production \
    /backup/k8s-cluster/namespaces/staging \
    /backup/k8s-cluster/namespaces/default \
    /backup/k8s-cluster/deployments \
    /backup/k8s-cluster/configmaps \
    /solution

COPY data/backup/k8s-cluster/namespaces/production/secrets.yaml /backup/k8s-cluster/namespaces/production/secrets.yaml
COPY data/backup/k8s-cluster/namespaces/staging/db-secrets.yaml /backup/k8s-cluster/namespaces/staging/db-secrets.yaml
COPY data/backup/k8s-cluster/namespaces/default/secrets.yaml /backup/k8s-cluster/namespaces/default/secrets.yaml
COPY data/backup/k8s-cluster/deployments/production-app.yaml /backup/k8s-cluster/deployments/production-app.yaml
COPY data/backup/k8s-cluster/namespaces/production/app-secrets.yaml /backup/k8s-cluster/namespaces/production/app-secrets.yaml
COPY data/backup/k8s-cluster/namespaces/staging/app-config.yaml /backup/k8s-cluster/namespaces/staging/app-config.yaml
COPY data/backup/k8s-cluster/configmaps/production-config.yaml /backup/k8s-cluster/configmaps/production-config.yaml
COPY data/backup/k8s-cluster/namespaces/default/db-init-secret.yaml /backup/k8s-cluster/namespaces/default/db-init-secret.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /backup