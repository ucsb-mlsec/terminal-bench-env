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
    apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/vulkan_src

COPY vulkan_src/buffer_manager.cpp /workspace/vulkan_src/buffer_manager.cpp
COPY vulkan_src/command_recorder.cpp /workspace/vulkan_src/command_recorder.cpp
COPY vulkan_src/descriptor_binder.cpp /workspace/vulkan_src/descriptor_binder.cpp
COPY vulkan_src/image_processor.cpp /workspace/vulkan_src/image_processor.cpp
COPY vulkan_src/memory_allocator.cpp /workspace/vulkan_src/memory_allocator.cpp
COPY vulkan_src/pipeline_builder.cpp /workspace/vulkan_src/pipeline_builder.cpp
COPY vulkan_src/render_pass_manager.cpp /workspace/vulkan_src/render_pass_manager.cpp
COPY vulkan_src/resource_destroyer.cpp /workspace/vulkan_src/resource_destroyer.cpp
COPY vulkan_src/shader_loader.cpp /workspace/vulkan_src/shader_loader.cpp
COPY vulkan_src/swapchain_setup.cpp /workspace/vulkan_src/swapchain_setup.cpp
COPY vulkan_src/sync_primitives.cpp /workspace/vulkan_src/sync_primitives.cpp
COPY vulkan_src/texture_sampler.cpp /workspace/vulkan_src/texture_sampler.cpp
COPY vulkan_spec.txt /workspace/vulkan_spec.txt

WORKDIR /workspace