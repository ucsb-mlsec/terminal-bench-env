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
    apt-get install -y openjdk-17-jdk curl unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/java-app/gradle/wrapper /workspace/java-app/src/main/java/com/example /workspace/java-app/src/test/java/com/example

COPY java-app/build.gradle /workspace/java-app/build.gradle
COPY java-app/settings.gradle /workspace/java-app/settings.gradle
COPY java-app/src/main/java/com/example/App.java /workspace/java-app/src/main/java/com/example/App.java
COPY java-app/src/main/java/com/example/Calculator.java /workspace/java-app/src/main/java/com/example/Calculator.java
COPY java-app/src/test/java/com/example/CalculatorTest.java /workspace/java-app/src/test/java/com/example/CalculatorTest.java
COPY java-app/gradle/wrapper/gradle-wrapper.properties /workspace/java-app/gradle/wrapper/gradle-wrapper.properties

RUN cd /workspace/java-app && \
    curl -L https://services.gradle.org/distributions/gradle-8.5-bin.zip -o gradle.zip && \
    unzip -q gradle.zip && \
    mv gradle-8.5 /opt/gradle && \
    rm gradle.zip && \
    ln -s /opt/gradle/bin/gradle /usr/bin/gradle

ENV GRADLE_HOME=/opt/gradle
ENV PATH=$PATH:$GRADLE_HOME/bin

WORKDIR /workspace