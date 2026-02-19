FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/proofs /workspace/solution

COPY proofs/proof_01.txt /workspace/proofs/proof_01.txt
COPY proofs/proof_02.txt /workspace/proofs/proof_02.txt
COPY proofs/proof_03.txt /workspace/proofs/proof_03.txt
COPY proofs/proof_04.txt /workspace/proofs/proof_04.txt
COPY proofs/proof_05.txt /workspace/proofs/proof_05.txt

WORKDIR /workspace