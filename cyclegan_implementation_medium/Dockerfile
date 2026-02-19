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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/cyclegan_configs /workspace/datasets/horses2zebras /workspace/datasets/cityscapes

COPY config/config1.yaml /workspace/cyclegan_configs/config1.yaml
COPY config/config2.yaml /workspace/cyclegan_configs/config2.yaml
COPY config/config3.yaml /workspace/cyclegan_configs/config3.yaml
COPY config/config4.yaml /workspace/cyclegan_configs/config4.yaml
COPY config/config5.yaml /workspace/cyclegan_configs/config5.yaml
COPY config/config6.yaml /workspace/cyclegan_configs/config6.yaml
COPY config/config7.yaml /workspace/cyclegan_configs/config7.yaml
COPY config/config8.yml /workspace/cyclegan_configs/config8.yml
COPY config/config9.yaml /workspace/cyclegan_configs/config9.yaml
COPY config/config10.yml /workspace/cyclegan_configs/config10.yml
COPY data/datasets/horses2zebras/.gitkeep /workspace/datasets/horses2zebras/.gitkeep
COPY data/datasets/cityscapes/.gitkeep /workspace/datasets/cityscapes/.gitkeep

WORKDIR /workspace