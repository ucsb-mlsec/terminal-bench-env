FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Coq and OCaml dependencies
RUN apt-get update && apt-get install -y \
    coq \
    ocaml \
    ocaml-native-compilers \
    ocamlbuild \
    opam \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y ocaml coq && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/coq_project/src \
    /workspace/coq_project/extracted

# Copy Coq source files
COPY coq_project/src/DataStructures.v /workspace/coq_project/src/DataStructures.v
COPY coq_project/src/Extraction.v /workspace/coq_project/src/Extraction.v

# Copy extracted OCaml files
COPY coq_project/extracted/DataStructures.ml /workspace/coq_project/extracted/DataStructures.ml
COPY coq_project/extracted/DataStructures.mli /workspace/coq_project/extracted/DataStructures.mli

WORKDIR /workspace