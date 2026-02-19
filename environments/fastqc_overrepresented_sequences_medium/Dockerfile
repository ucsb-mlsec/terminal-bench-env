FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages beautifulsoup4

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/fastqc_reports /solution

COPY data/fastqc_reports/sample1_fastqc.txt /data/fastqc_reports/sample1_fastqc.txt
COPY data/fastqc_reports/sample2_fastqc.txt /data/fastqc_reports/sample2_fastqc.txt
COPY data/fastqc_reports/sample3_fastqc.txt /data/fastqc_reports/sample3_fastqc.txt
COPY data/fastqc_reports/sample4_fastqc.txt /data/fastqc_reports/sample4_fastqc.txt
COPY data/fastqc_reports/sample5_fastqc.txt /data/fastqc_reports/sample5_fastqc.txt
COPY data/fastqc_reports/sample6_fastqc.txt /data/fastqc_reports/sample6_fastqc.txt
COPY data/fastqc_reports/sample7_fastqc.txt /data/fastqc_reports/sample7_fastqc.txt
COPY data/fastqc_reports/sample8_fastqc.txt /data/fastqc_reports/sample8_fastqc.txt

WORKDIR /workspace