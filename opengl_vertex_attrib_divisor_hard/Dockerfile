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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /opt/renderer_analysis/config /opt/renderer_analysis/data

COPY renderer_sim.py /opt/renderer_analysis/renderer_sim.py
COPY config/attribute_config.yaml /opt/renderer_analysis/attribute_config.yaml
COPY data/vertex_data.txt /opt/renderer_analysis/data/vertex_data.txt
COPY data/instance_data.txt /opt/renderer_analysis/data/instance_data.txt
COPY data/render_artifacts.log /opt/renderer_analysis/data/render_artifacts.log

WORKDIR /opt/renderer_analysis