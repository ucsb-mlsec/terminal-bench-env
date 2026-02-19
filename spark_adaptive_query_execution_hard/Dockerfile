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
    apt-get install -y --no-install-recommends \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bash && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/spark/conf \
    /var/log/spark \
    /usr/local/bin \
    /home/agent

COPY config/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
COPY data/query-execution.log /var/log/spark/query-execution.log
COPY validate-spark-config.sh /usr/local/bin/validate-spark-config.sh

RUN chmod +x /usr/local/bin/validate-spark-config.sh

WORKDIR /home/agent