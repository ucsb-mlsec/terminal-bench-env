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
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy pillow

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    pillow

RUN mkdir -p /workspace/renderer/src \
    /workspace/renderer/test_scenes \
    /workspace/renderer/output

COPY data/reference_output.png /workspace/renderer/test_scenes/reference_output.png
COPY pipeline.py /workspace/renderer/src/pipeline.py
COPY scene_loader.py /workspace/renderer/src/scene_loader.py
COPY vertex_shader.py /workspace/renderer/src/vertex_shader.py
COPY fragment_shader.py /workspace/renderer/src/fragment_shader.py
COPY output_generator.py /workspace/renderer/src/output_generator.py

WORKDIR /workspace