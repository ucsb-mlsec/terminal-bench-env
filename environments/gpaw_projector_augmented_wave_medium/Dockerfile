FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/gpaw-datasets /workspace

COPY data/H.PBE.gz /opt/gpaw-datasets/H.PBE.gz
COPY data/H.LDA.gz /opt/gpaw-datasets/H.LDA.gz
COPY data/C.PBE.gz /opt/gpaw-datasets/C.PBE.gz
COPY data/O.PBE.gz /opt/gpaw-datasets/O.PBE.gz
COPY data/O.LDA.gz /opt/gpaw-datasets/O.LDA.gz
COPY data/S.PW91.gz /opt/gpaw-datasets/S.PW91.gz
COPY data/P.PBE.gz /opt/gpaw-datasets/P.PBE.gz
COPY data/Si.LDA.gz /opt/gpaw-datasets/Si.LDA.gz
COPY data/Fe.PBE.gz /opt/gpaw-datasets/Fe.PBE.gz
COPY data/Al.PBE.gz /opt/gpaw-datasets/Al.PBE.gz
COPY data/Na.LDA.gz /opt/gpaw-datasets/Na.LDA.gz
COPY data/Mg.PBE.gz /opt/gpaw-datasets/Mg.PBE.gz

WORKDIR /workspace