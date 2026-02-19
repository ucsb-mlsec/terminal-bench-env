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

COPY shaders/vertex_basic.glsl /workspace/shaders/vertex_basic.glsl
COPY shaders/fragment_gather.frag /workspace/shaders/fragment_gather.frag
COPY shaders/fma_compute.glsl /workspace/shaders/fma_compute.glsl
COPY shaders/multipass_deferred.frag /workspace/shaders/multipass_deferred.frag
COPY shaders/lighting_pass.frag /workspace/shaders/lighting_pass.frag
COPY shaders/precise_unused.glsl /workspace/shaders/precise_unused.glsl
COPY shaders/complex_gather.frag /workspace/shaders/complex_gather.frag
COPY shaders/dynamic_sampler.frag /workspace/shaders/dynamic_sampler.frag
COPY shaders/fma_mixed.frag /workspace/shaders/fma_mixed.frag
COPY shaders/postprocess.frag /workspace/shaders/postprocess.frag

WORKDIR /workspace