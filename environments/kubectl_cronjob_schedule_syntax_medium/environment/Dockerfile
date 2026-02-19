FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /home/user/k8s-manifests /home/user/fixed-manifests

COPY config/cronjob1.yaml /home/user/k8s-manifests/cronjob1.yaml
COPY config/cronjob2.yaml /home/user/k8s-manifests/cronjob2.yaml
COPY config/cronjob3.yaml /home/user/k8s-manifests/cronjob3.yaml
COPY config/cronjob4.yaml /home/user/k8s-manifests/cronjob4.yaml
COPY config/cronjob5.yaml /home/user/k8s-manifests/cronjob5.yaml
COPY config/cronjob6.yaml /home/user/k8s-manifests/cronjob6.yaml
COPY config/cronjob7.yaml /home/user/k8s-manifests/cronjob7.yaml
COPY config/cronjob8.yaml /home/user/k8s-manifests/cronjob8.yaml
COPY config/cronjob9.yaml /home/user/k8s-manifests/cronjob9.yaml
COPY config/cronjob10.yaml /home/user/k8s-manifests/cronjob10.yaml
COPY config/deployment.yaml /home/user/k8s-manifests/deployment.yaml

WORKDIR /home/user

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
