FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/protein /workspace/ligands /workspace/solution

COPY data/protein/binding_site.csv /workspace/protein/binding_site.csv
COPY data/ligands/ligand_001.csv /workspace/ligands/ligand_001.csv
COPY data/ligands/ligand_002.csv /workspace/ligands/ligand_002.csv
COPY data/ligands/ligand_003.csv /workspace/ligands/ligand_003.csv
COPY data/ligands/ligand_004.csv /workspace/ligands/ligand_004.csv
COPY data/ligands/ligand_005.csv /workspace/ligands/ligand_005.csv
COPY data/ligands/ligand_006.csv /workspace/ligands/ligand_006.csv
COPY data/ligands/ligand_007.csv /workspace/ligands/ligand_007.csv
COPY data/ligands/ligand_008.csv /workspace/ligands/ligand_008.csv

RUN pip3 install --no-cache-dir --break-system-packages numpy pandas

WORKDIR /workspace