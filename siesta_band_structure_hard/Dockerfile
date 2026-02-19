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

RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN mkdir -p /workspace/data

COPY data/silicon.txt /workspace/data/silicon.txt
COPY data/gaas.txt /workspace/data/gaas.txt
COPY data/graphene.txt /workspace/data/graphene.txt
COPY data/diamond.txt /workspace/data/diamond.txt
COPY data/inas.txt /workspace/data/inas.txt
COPY data/corrupted_material.txt /workspace/data/corrupted_material.txt
COPY analyze_bands.py /workspace/analyze_bands.py

WORKDIR /workspace