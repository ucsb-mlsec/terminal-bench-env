FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace /workspace/config

COPY config/spark-defaults.conf /workspace/spark-defaults.conf
COPY config/core-site.xml /workspace/core-site.xml
COPY config/application.properties /workspace/application.properties
COPY REQUIREMENTS.md /workspace/REQUIREMENTS.md
COPY committer-options.txt /workspace/committer-options.txt

WORKDIR /workspace