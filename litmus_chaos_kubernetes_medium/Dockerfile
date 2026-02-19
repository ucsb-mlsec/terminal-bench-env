FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/chaos-manifests

COPY chaos-manifests/valid-pod-cpu-hog.yaml /workspace/chaos-manifests/valid-pod-cpu-hog.yaml
COPY chaos-manifests/valid-pod-network-loss.yaml /workspace/chaos-manifests/valid-pod-network-loss.yaml
COPY chaos-manifests/valid-container-kill.yaml /workspace/chaos-manifests/valid-container-kill.yaml
COPY chaos-manifests/invalid-missing-apiversion.yaml /workspace/chaos-manifests/invalid-missing-apiversion.yaml
COPY chaos-manifests/invalid-wrong-kind.yaml /workspace/chaos-manifests/invalid-wrong-kind.yaml
COPY chaos-manifests/invalid-missing-metadata-name.yaml /workspace/chaos-manifests/invalid-missing-metadata-name.yaml
COPY chaos-manifests/invalid-missing-image.yaml /workspace/chaos-manifests/invalid-missing-image.yaml
COPY chaos-manifests/invalid-missing-args.yaml /workspace/chaos-manifests/invalid-missing-args.yaml
COPY chaos-manifests/invalid-missing-env.yaml /workspace/chaos-manifests/invalid-missing-env.yaml
COPY chaos-manifests/invalid-malformed-yaml.yaml /workspace/chaos-manifests/invalid-malformed-yaml.yaml

WORKDIR /workspace