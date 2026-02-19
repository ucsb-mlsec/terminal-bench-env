FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/genomics/modules \
    /opt/genomics/lib \
    /usr/local/lib \
    /home/genomics/utils \
    /var/genomics/analysis \
    /opt/analysis_tools \
    /solution

COPY opt/genomics/analyze_sequences.py /opt/genomics/analyze_sequences.py
COPY opt/genomics/modules/sequence_processor.py /opt/genomics/modules/sequence_processor.py
COPY usr/local/lib/genomic_stats.py /usr/local/lib/genomic_stats.py
COPY home/genomics/utils/data_formatter.py /home/genomics/utils/data_formatter.py
COPY var/genomics/analysis/quality_control.py /var/genomics/analysis/quality_control.py
COPY opt/analysis_tools/variant_caller.py /opt/analysis_tools/variant_caller.py
COPY opt/genomics/lib/alignment_tools.py /opt/genomics/lib/alignment_tools.py

WORKDIR /opt/genomics