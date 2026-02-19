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

RUN mkdir -p /workspace/k8s-config

COPY config/deployment.yaml /workspace/k8s-config/deployment.yaml
COPY config/configmap.yaml /workspace/k8s-config/configmap.yaml
COPY app.py /workspace/k8s-config/app.py
COPY requirements.txt /workspace/k8s-config/requirements.txt

RUN if [ -s /workspace/k8s-config/requirements.txt ]; then \
    pip3 install --no-cache-dir --break-system-packages -r /workspace/k8s-config/requirements.txt; \
    fi

WORKDIR /workspace