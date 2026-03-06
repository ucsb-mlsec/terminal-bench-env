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
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/shader_analysis

COPY particle_update.vert /home/user/shader_analysis/particle_update.vert
COPY mesh_transform.vert /home/user/shader_analysis/mesh_transform.vert
COPY terrain_generator.geom /home/user/shader_analysis/terrain_generator.geom
COPY water_simulation.vert /home/user/shader_analysis/water_simulation.vert
COPY main.cpp /home/user/shader_analysis/main.cpp
COPY shaders.conf /home/user/shader_analysis/shaders.conf

WORKDIR /home/user/shader_analysis

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
