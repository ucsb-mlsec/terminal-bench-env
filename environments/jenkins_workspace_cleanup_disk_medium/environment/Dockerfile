FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/lib/jenkins/workspace/workspace1 \
    /var/lib/jenkins/workspace/workspace2/artifacts \
    /var/lib/jenkins/workspace/workspace3 \
    /var/lib/jenkins/workspace/workspace4 \
    /var/lib/jenkins/workspace/workspace5 \
    /var/lib/jenkins/workspace/workspace6 \
    /var/lib/jenkins/workspace/workspace7/cache \
    /var/lib/jenkins/workspace/workspace8

COPY data/workspace1/build.log /var/lib/jenkins/workspace/workspace1/build.log
COPY data/workspace1/artifact.jar /var/lib/jenkins/workspace/workspace1/artifact.jar
COPY data/workspace2/build.log /var/lib/jenkins/workspace/workspace2/build.log
COPY data/workspace2/artifacts/large_binary.bin /var/lib/jenkins/workspace/workspace2/artifacts/large_binary.bin
COPY data/workspace3/build.log /var/lib/jenkins/workspace/workspace3/build.log
COPY data/workspace3/temp_data.bin /var/lib/jenkins/workspace/workspace3/temp_data.bin
COPY data/workspace4/build.log /var/lib/jenkins/workspace/workspace4/build.log
COPY data/workspace4/output.dat /var/lib/jenkins/workspace/workspace4/output.dat
COPY data/workspace5/build.log /var/lib/jenkins/workspace/workspace5/build.log
COPY data/workspace5/small_artifact.jar /var/lib/jenkins/workspace/workspace5/small_artifact.jar
COPY data/workspace6/build.log /var/lib/jenkins/workspace/workspace6/build.log
COPY data/workspace6/deploy_package.tar.gz /var/lib/jenkins/workspace/workspace6/deploy_package.tar.gz
COPY data/workspace7/build.log /var/lib/jenkins/workspace/workspace7/build.log
COPY data/workspace7/cache/data.cache /var/lib/jenkins/workspace/workspace7/cache/data.cache
COPY data/workspace8/build.log /var/lib/jenkins/workspace/workspace8/build.log
COPY data/workspace8/result.txt /var/lib/jenkins/workspace/workspace8/result.txt

WORKDIR /var/lib/jenkins/workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
