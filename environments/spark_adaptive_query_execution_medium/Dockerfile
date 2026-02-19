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
    openjdk-11-jdk-headless \
    wget \
    tar && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz && \
    tar -xzf spark-3.3.0-bin-hadoop3.tgz -C /opt/ && \
    mv /opt/spark-3.3.0-bin-hadoop3 /opt/spark && \
    rm spark-3.3.0-bin-hadoop3.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

RUN mkdir -p /opt/spark/conf /workspace/queries /workspace

COPY data/queries/analytical_query1.sql /workspace/queries/analytical_query1.sql
COPY data/queries/analytical_query2.sql /workspace/queries/analytical_query2.sql
COPY data/queries/skewed_join_query.sql /workspace/queries/skewed_join_query.sql
COPY config/spark-defaults.conf /opt/spark/conf/spark-defaults.conf

WORKDIR /workspace