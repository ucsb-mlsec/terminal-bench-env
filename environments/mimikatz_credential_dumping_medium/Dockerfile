FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/evidence /workspace/solution

COPY data/evidence/process_dump.log /workspace/evidence/process_dump.log
COPY data/evidence/registry_extract.log /workspace/evidence/registry_extract.log
COPY data/evidence/network_capture.log /workspace/evidence/network_capture.log
COPY data/evidence/browser_artifacts.log /workspace/evidence/browser_artifacts.log
COPY data/evidence/lsass_strings.log /workspace/evidence/lsass_strings.log
COPY data/evidence/mimikatz_output.txt /workspace/evidence/mimikatz_output.txt

WORKDIR /workspace