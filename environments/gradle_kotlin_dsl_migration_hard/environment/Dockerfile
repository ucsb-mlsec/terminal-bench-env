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

RUN mkdir -p /workspace/src/main/java/com/example && \
    mkdir -p /workspace/src/test/java/com/example && \
    mkdir -p /workspace/gradle/wrapper

WORKDIR /workspace

COPY settings.gradle /workspace/settings.gradle
COPY build.gradle /workspace/build.gradle
COPY src/main/java/com/example/StringUtils.java /workspace/src/main/java/com/example/StringUtils.java
COPY src/test/java/com/example/StringUtilsTest.java /workspace/src/test/java/com/example/StringUtilsTest.java
COPY gradlew /workspace/gradlew
COPY gradlew.bat /workspace/gradlew.bat
COPY gradle/wrapper/gradle-wrapper.properties /workspace/gradle/wrapper/gradle-wrapper.properties
COPY gradle/wrapper/gradle-wrapper.jar /workspace/gradle/wrapper/gradle-wrapper.jar

RUN chmod +x /workspace/gradlew

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
