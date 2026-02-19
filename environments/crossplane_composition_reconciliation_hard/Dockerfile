FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/crossplane-configs

COPY crossplane-configs/xrd.yaml /workspace/crossplane-configs/xrd.yaml
COPY crossplane-configs/composition.yaml /workspace/crossplane-configs/composition.yaml
COPY crossplane-configs/claim.yaml /workspace/crossplane-configs/claim.yaml
COPY crossplane-configs/provider-config.yaml /workspace/crossplane-configs/provider-config.yaml
COPY crossplane-configs/controller.log /workspace/crossplane-configs/controller.log

WORKDIR /workspace