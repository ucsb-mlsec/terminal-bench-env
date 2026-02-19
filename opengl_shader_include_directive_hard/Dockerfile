FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders/common /workspace/shaders/lighting /workspace/shaders/effects /workspace/shaders/advanced /workspace/shaders/particles /workspace/shaders/shadow

COPY shaders/main.glsl /workspace/shaders/main.glsl
COPY shaders/vertex.glsl /workspace/shaders/vertex.glsl
COPY shaders/fragment.glsl /workspace/shaders/fragment.glsl
COPY shaders/compute.glsl /workspace/shaders/compute.glsl
COPY shaders/common/utils.glsl /workspace/shaders/common/utils.glsl
COPY shaders/common/defines.glsl /workspace/shaders/common/defines.glsl
COPY shaders/lighting/phong.glsl /workspace/shaders/lighting/phong.glsl
COPY shaders/effects/blur.glsl /workspace/shaders/effects/blur.glsl
COPY shaders/effects/bloom.glsl /workspace/shaders/effects/bloom.glsl
COPY shaders/post_process.glsl /workspace/shaders/post_process.glsl
COPY shaders/advanced/deferred.glsl /workspace/shaders/advanced/deferred.glsl
COPY shaders/particles/particle.glsl /workspace/shaders/particles/particle.glsl
COPY shaders/particles/emitter.glsl /workspace/shaders/particles/emitter.glsl
COPY shaders/shadow/shadow_map.glsl /workspace/shaders/shadow/shadow_map.glsl

WORKDIR /workspace