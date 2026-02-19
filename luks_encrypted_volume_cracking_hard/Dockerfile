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
    apt-get install -y \
    hashcat \
    john \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages bcrypt

RUN mkdir -p /home/agent/intel /home/agent/wordlists

COPY data/recovered_hashes.txt /home/agent/recovered_hashes.txt
COPY data/intel/pattern_analysis.txt /home/agent/intel/pattern_analysis.txt
COPY data/intel/operational_notes.txt /home/agent/intel/operational_notes.txt
COPY data/wordlists/base_words.txt /home/agent/wordlists/base_words.txt

WORKDIR /home/agent