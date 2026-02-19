FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sim_setup

COPY sim_setup/system.gro /workspace/sim_setup/system.gro
COPY sim_setup/system.top /workspace/sim_setup/system.top
COPY sim_setup/protein.itp /workspace/sim_setup/protein.itp
COPY sim_setup/ligand.itp /workspace/sim_setup/ligand.itp
COPY sim_setup/forcefield.itp /workspace/sim_setup/forcefield.itp

WORKDIR /workspace