FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders

COPY shaders/lighting_basic.frag /workspace/shaders/lighting_basic.frag
COPY shaders/pbr_lighting.frag /workspace/shaders/pbr_lighting.frag
COPY shaders/color_correction.frag /workspace/shaders/color_correction.frag
COPY shaders/bloom_combine.frag /workspace/shaders/bloom_combine.frag
COPY shaders/tone_mapping.frag /workspace/shaders/tone_mapping.frag
COPY shaders/shadow_mapping.frag /workspace/shaders/shadow_mapping.frag
COPY shaders/ambient_occlusion.frag /workspace/shaders/ambient_occlusion.frag
COPY shaders/reflection_probe.frag /workspace/shaders/reflection_probe.frag
COPY shaders/gaussian_blur.frag /workspace/shaders/gaussian_blur.frag
COPY shaders/deferred_combine.frag /workspace/shaders/deferred_combine.frag

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
