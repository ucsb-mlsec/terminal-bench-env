FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/network_data /home/user/solution

COPY data/router_graph.json /home/user/network_data/router_graph.json
COPY data/mesh_network.json /home/user/network_data/mesh_network.json
COPY data/star_topology.json /home/user/network_data/star_topology.json
COPY data/ring_network.json /home/user/network_data/ring_network.json
COPY data/complex_graph.json /home/user/network_data/complex_graph.json

WORKDIR /home/user