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
    apt-get install -y wget openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://archive.apache.org/dist/flink/flink-1.17.1/flink-1.17.1-bin-scala_2.12.tgz && \
    tar -xzf flink-1.17.1-bin-scala_2.12.tgz && \
    mv flink-1.17.1 /opt/flink && \
    rm flink-1.17.1-bin-scala_2.12.tgz

ENV FLINK_HOME=/opt/flink
ENV PATH=$FLINK_HOME/bin:$PATH

RUN mkdir -p /data /solution

COPY data/purchases.csv /data/purchases.csv

WORKDIR /workspace