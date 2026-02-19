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
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/processor-src/META-INF/services && \
    mkdir -p /workspace/app-src && \
    mkdir -p /workspace/output/classes

COPY processor-src/AutoGetter.java /workspace/processor-src/AutoGetter.java
COPY processor-src/AutoGetterProcessor.java /workspace/processor-src/AutoGetterProcessor.java
COPY processor-src/META-INF/services/javax.annotation.processing.Processor /workspace/processor-src/META-INF/services/javax.annotation.processing.Processor
COPY app-src/Person.java /workspace/app-src/Person.java

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
