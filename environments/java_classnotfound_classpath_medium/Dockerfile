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
    apt-get install -y openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/biotools/sequence-analyzer /opt/biotools/lib /data

COPY opt/biotools/sequence-analyzer/seqanalyzer.jar /opt/biotools/sequence-analyzer/seqanalyzer.jar
COPY opt/biotools/sequence-analyzer/run.sh /opt/biotools/sequence-analyzer/run.sh
COPY data/test_sequence.fasta /opt/biotools/sequence-analyzer/test_sequence.fasta
COPY opt/biotools/lib/commons-io-2.11.0.jar /opt/biotools/lib/commons-io-2.11.0.jar
COPY opt/biotools/lib/commons-lang3-3.12.0.jar /opt/biotools/lib/commons-lang3-3.12.0.jar

WORKDIR /opt/biotools/sequence-analyzer