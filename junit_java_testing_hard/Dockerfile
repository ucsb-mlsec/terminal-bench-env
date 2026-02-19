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
    apt-get install -y maven openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN mkdir -p /workspace/validator-service/src/main/java/com/example/validator && \
    mkdir -p /workspace/validator-service/src/test/java/com/example/validator

COPY workspace/validator-service/pom.xml /workspace/validator-service/pom.xml
COPY workspace/validator-service/src/main/java/com/example/validator/DataValidator.java /workspace/validator-service/src/main/java/com/example/validator/DataValidator.java
COPY workspace/validator-service/src/main/java/com/example/validator/RuleEngine.java /workspace/validator-service/src/main/java/com/example/validator/RuleEngine.java
COPY workspace/validator-service/src/test/java/com/example/validator/DataValidatorTest.java /workspace/validator-service/src/test/java/com/example/validator/DataValidatorTest.java
COPY workspace/validator-service/src/test/java/com/example/validator/RuleEngineTest.java /workspace/validator-service/src/test/java/com/example/validator/RuleEngineTest.java

WORKDIR /workspace/validator-service