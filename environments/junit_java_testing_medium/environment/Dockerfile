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
    apt-get install -y maven openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN mkdir -p /workspace/src/main/java/com/userservice && \
    mkdir -p /workspace/src/test/java/com/userservice

COPY pom.xml /workspace/pom.xml
COPY src/main/java/com/userservice/User.java /workspace/src/main/java/com/userservice/User.java
COPY src/main/java/com/userservice/UserService.java /workspace/src/main/java/com/userservice/UserService.java
COPY src/main/java/com/userservice/UserValidator.java /workspace/src/main/java/com/userservice/UserValidator.java
COPY src/test/java/com/userservice/UserServiceTest.java /workspace/src/test/java/com/userservice/UserServiceTest.java
COPY src/test/java/com/userservice/UserValidatorTest.java /workspace/src/test/java/com/userservice/UserValidatorTest.java

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
