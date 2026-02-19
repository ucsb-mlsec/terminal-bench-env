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

RUN pip3 install --no-cache-dir --break-system-packages networkx

RUN mkdir -p /data/molecules /output

COPY data/molecules/molecule_001.edges /data/molecules/molecule_001.edges
COPY data/molecules/molecule_001.atoms /data/molecules/molecule_001.atoms
COPY data/molecules/molecule_002.edges /data/molecules/molecule_002.edges
COPY data/molecules/molecule_002.atoms /data/molecules/molecule_002.atoms
COPY data/molecules/molecule_003.edges /data/molecules/molecule_003.edges
COPY data/molecules/molecule_003.atoms /data/molecules/molecule_003.atoms
COPY data/molecules/molecule_004.edges /data/molecules/molecule_004.edges
COPY data/molecules/molecule_004.atoms /data/molecules/molecule_004.atoms
COPY data/molecules/molecule_005.edges /data/molecules/molecule_005.edges
COPY data/molecules/molecule_005.atoms /data/molecules/molecule_005.atoms
COPY data/molecules/molecule_006.edges /data/molecules/molecule_006.edges
COPY data/molecules/molecule_006.atoms /data/molecules/molecule_006.atoms
COPY data/molecules/molecule_007.edges /data/molecules/molecule_007.edges
COPY data/molecules/molecule_007.atoms /data/molecules/molecule_007.atoms
COPY data/molecules/molecule_008.edges /data/molecules/molecule_008.edges
COPY data/molecules/molecule_008.atoms /data/molecules/molecule_008.atoms
COPY data/molecules/molecule_009.edges /data/molecules/molecule_009.edges
COPY data/molecules/molecule_009.atoms /data/molecules/molecule_009.atoms
COPY data/molecules/molecule_010.edges /data/molecules/molecule_010.edges
COPY data/molecules/molecule_010.atoms /data/molecules/molecule_010.atoms

WORKDIR /workspace