FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /config /solution

COPY data/genomics-parser.log /data/genomics-parser.log
COPY data/sequence-aligner.log /data/sequence-aligner.log
COPY data/variant-caller.log /data/variant-caller.log
COPY config/genomics-parser.conf /config/genomics-parser.conf
COPY config/sequence-aligner.conf /config/sequence-aligner.conf
COPY config/variant-caller.conf /config/variant-caller.conf
COPY config/system_resources.txt /config/system_resources.txt
COPY start-genomics-parser.sh /start-genomics-parser.sh
COPY start-sequence-aligner.sh /start-sequence-aligner.sh
COPY start-variant-caller.sh /start-variant-caller.sh

RUN chmod +x /start-genomics-parser.sh /start-sequence-aligner.sh /start-variant-caller.sh

WORKDIR /workspace