FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/vasp_runs/encut_200 && \
    mkdir -p /workspace/vasp_runs/encut_250 && \
    mkdir -p /workspace/vasp_runs/encut_300 && \
    mkdir -p /workspace/vasp_runs/encut_350 && \
    mkdir -p /workspace/vasp_runs/encut_400 && \
    mkdir -p /workspace/vasp_runs/encut_450 && \
    mkdir -p /workspace/vasp_runs/encut_500 && \
    mkdir -p /workspace/vasp_runs/encut_550 && \
    mkdir -p /workspace/vasp_runs/encut_600

COPY data/vasp_runs/encut_200/OUTCAR /workspace/vasp_runs/encut_200/OUTCAR
COPY data/vasp_runs/encut_250/OUTCAR /workspace/vasp_runs/encut_250/OUTCAR
COPY data/vasp_runs/encut_300/OUTCAR /workspace/vasp_runs/encut_300/OUTCAR
COPY data/vasp_runs/encut_350/OUTCAR /workspace/vasp_runs/encut_350/OUTCAR
COPY data/vasp_runs/encut_400/OUTCAR /workspace/vasp_runs/encut_400/OUTCAR
COPY data/vasp_runs/encut_450/OUTCAR /workspace/vasp_runs/encut_450/OUTCAR
COPY data/vasp_runs/encut_500/OUTCAR /workspace/vasp_runs/encut_500/OUTCAR
COPY data/vasp_runs/encut_550/OUTCAR /workspace/vasp_runs/encut_550/OUTCAR
COPY data/vasp_runs/encut_600/OUTCAR /workspace/vasp_runs/encut_600/OUTCAR

WORKDIR /workspace