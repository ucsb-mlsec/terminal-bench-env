FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/chipseq_metrics /output

COPY data/chipseq_metrics/H3K4me3_rep1.txt /data/chipseq_metrics/H3K4me3_rep1.txt
COPY data/chipseq_metrics/H3K4me3_rep2.txt /data/chipseq_metrics/H3K4me3_rep2.txt
COPY data/chipseq_metrics/H3K4me3_rep3.txt /data/chipseq_metrics/H3K4me3_rep3.txt
COPY data/chipseq_metrics/H3K27ac_rep1.txt /data/chipseq_metrics/H3K27ac_rep1.txt
COPY data/chipseq_metrics/H3K27ac_rep2.txt /data/chipseq_metrics/H3K27ac_rep2.txt
COPY data/chipseq_metrics/H3K27ac_rep3.txt /data/chipseq_metrics/H3K27ac_rep3.txt
COPY data/chipseq_metrics/H3K36me3_rep1.txt /data/chipseq_metrics/H3K36me3_rep1.txt
COPY data/chipseq_metrics/H3K36me3_rep2.txt /data/chipseq_metrics/H3K36me3_rep2.txt
COPY data/chipseq_metrics/H3K36me3_rep3.txt /data/chipseq_metrics/H3K36me3_rep3.txt
COPY data/chipseq_metrics/H3K9me3_rep1.txt /data/chipseq_metrics/H3K9me3_rep1.txt
COPY data/chipseq_metrics/H3K9me3_rep2.txt /data/chipseq_metrics/H3K9me3_rep2.txt
COPY data/chipseq_metrics/H3K9me3_rep3.txt /data/chipseq_metrics/H3K9me3_rep3.txt

WORKDIR /workspace