FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /k8s-manifests /solution

COPY k8s-manifests/frontend-deployment.yaml /k8s-manifests/frontend-deployment.yaml
COPY k8s-manifests/frontend-service.yaml /k8s-manifests/frontend-service.yaml
COPY k8s-manifests/api-gateway-deployment.yaml /k8s-manifests/api-gateway-deployment.yaml
COPY k8s-manifests/api-gateway-service.yaml /k8s-manifests/api-gateway-service.yaml
COPY k8s-manifests/processing-service-deployment.yaml /k8s-manifests/processing-service-deployment.yaml
COPY k8s-manifests/processing-service-svc.yaml /k8s-manifests/processing-service-svc.yaml
COPY k8s-manifests/database-deployment.yaml /k8s-manifests/database-deployment.yaml
COPY k8s-manifests/database-service.yaml /k8s-manifests/database-service.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /workspace