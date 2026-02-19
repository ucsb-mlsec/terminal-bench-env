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
    apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN mkdir -p /workspace/gradle-app/src/main/java/com/example && \
    mkdir -p /workspace/gradle-app/src/test/java/com/example && \
    mkdir -p /workspace/gradle-app/gradle/wrapper

COPY gradle-app/build.gradle /workspace/gradle-app/
COPY gradle-app/gradle.properties /workspace/gradle-app/
COPY gradle-app/settings.gradle /workspace/gradle-app/
COPY gradle-app/gradlew /workspace/gradle-app/
COPY gradle-app/gradlew.bat /workspace/gradle-app/
COPY gradle-app/gradle/wrapper/gradle-wrapper.properties /workspace/gradle-app/gradle/wrapper/
COPY gradle-app/gradle/wrapper/gradle-wrapper.jar /workspace/gradle-app/gradle/wrapper/
COPY gradle-app/src/main/java/com/example/App.java /workspace/gradle-app/src/main/java/com/example/
COPY gradle-app/src/test/java/com/example/AppTest.java /workspace/gradle-app/src/test/java/com/example/

RUN chmod +x /workspace/gradle-app/gradlew

WORKDIR /workspace