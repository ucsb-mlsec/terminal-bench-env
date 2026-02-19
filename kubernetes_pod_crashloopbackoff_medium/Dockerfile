FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/k8s-app

COPY k8s-app/deployment.yaml /workspace/k8s-app/deployment.yaml
COPY k8s-app/service.yaml /workspace/k8s-app/service.yaml
COPY k8s-app/configmap.yaml /workspace/k8s-app/configmap.yaml
COPY k8s-app/app.py /workspace/k8s-app/app.py
COPY k8s-app/Dockerfile /workspace/k8s-app/Dockerfile
COPY k8s-app/requirements.txt /workspace/k8s-app/requirements.txt
COPY k8s-app/pod-logs.txt /workspace/k8s-app/pod-logs.txt

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /workspace