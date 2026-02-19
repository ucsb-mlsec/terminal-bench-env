FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/qchem_inputs /workspace/data

COPY qchem_inputs/benzene_template.in /workspace/qchem_inputs/benzene_template.in
COPY qchem_inputs/acetone_template.in /workspace/qchem_inputs/acetone_template.in
COPY qchem_inputs/methanol_template.in /workspace/qchem_inputs/methanol_template.in
COPY data/pcm_water_reference.txt /workspace/data/pcm_water_reference.txt
COPY data/benzene.xyz /workspace/data/benzene.xyz
COPY data/acetone.xyz /workspace/data/acetone.xyz
COPY data/methanol.xyz /workspace/data/methanol.xyz

WORKDIR /workspace