FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders/processing /workspace/shaders/effects /workspace/shaders/post /workspace/shaders/compute /workspace/shaders/lighting /workspace/effects

COPY BARRIER_RULES.md /workspace/
COPY shaders/blur.comp /workspace/shaders/
COPY shaders/processing/reduction.comp /workspace/shaders/processing/
COPY shaders/effects/particle.comp /workspace/shaders/effects/
COPY shaders/post/histogram.comp /workspace/shaders/post/
COPY shaders/compute/prefix_sum.comp /workspace/shaders/compute/
COPY shaders/lighting/deferred.comp /workspace/shaders/lighting/
COPY effects/raytracer.comp /workspace/effects/

WORKDIR /workspace