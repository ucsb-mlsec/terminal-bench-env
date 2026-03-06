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

RUN pip3 install --no-cache-dir --break-system-packages biopython

RUN mkdir -p /workspace/trees /workspace/output /workspace/output/nexus

COPY data/trees/species_a.nwk /workspace/trees/species_a.nwk
COPY data/trees/primates.txt /workspace/trees/primates.txt
COPY data/trees/mammals.tree /workspace/trees/mammals.tree
COPY data/trees/dataset_1.nexus /workspace/trees/dataset_1.nexus
COPY data/trees/analysis_output.nex /workspace/trees/analysis_output.nex
COPY data/trees/birds.nwk /workspace/trees/birds.nwk
COPY data/trees/corrupted_missing_semicolon.nwk /workspace/trees/corrupted_missing_semicolon.nwk
COPY data/trees/bad_parentheses.tree /workspace/trees/bad_parentheses.tree
COPY data/trees/invalid_chars.nwk /workspace/trees/invalid_chars.nwk
COPY data/trees/completely_broken.txt /workspace/trees/completely_broken.txt
COPY data/trees/misnamed_nexus.nwk /workspace/trees/misnamed_nexus.nwk
COPY data/trees/fish_phylogeny.tree /workspace/trees/fish_phylogeny.tree
COPY data/trees/plants.nexus /workspace/trees/plants.nexus
COPY data/trees/extra_spaces.nwk /workspace/trees/extra_spaces.nwk
COPY data/trees/insects.txt /workspace/trees/insects.txt
COPY data/trees/bacteria.nex /workspace/trees/bacteria.nex
COPY data/trees/reptiles.nwk /workspace/trees/reptiles.nwk
COPY data/trees/fungi.nexus /workspace/trees/fungi.nexus
COPY data/trees/amphibians.tree /workspace/trees/amphibians.tree

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
