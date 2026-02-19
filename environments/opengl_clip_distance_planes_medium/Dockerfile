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

COPY shaders/correct_basic.vert /workspace/shaders/correct_basic.vert
COPY shaders/correct_multiple_planes.vert /workspace/shaders/correct_multiple_planes.vert
COPY shaders/incorrect_declaration.vert /workspace/shaders/incorrect_declaration.vert
COPY shaders/incorrect_type_mismatch.vert /workspace/shaders/incorrect_type_mismatch.vert
COPY shaders/incorrect_array_bounds.vert /workspace/shaders/incorrect_array_bounds.vert
COPY shaders/incorrect_coordinate_space.vert /workspace/shaders/incorrect_coordinate_space.vert
COPY shaders/missing_output.vert /workspace/shaders/missing_output.vert
COPY shaders/correct_with_normals.vert /workspace/shaders/correct_with_normals.vert
COPY shaders/incorrect_plane_equation.vert /workspace/shaders/incorrect_plane_equation.vert
COPY shaders/correct_view_space.vert /workspace/shaders/correct_view_space.vert

WORKDIR /workspace