FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Java 17 and Maven
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk maven && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create necessary directories
RUN mkdir -p /workspace/src/main/java/com/example/auth && \
    mkdir -p /workspace/logs && \
    mkdir -p /tmp

# Copy Maven configuration
COPY pom.xml /workspace/pom.xml

# Copy Java source files
COPY src/main/java/com/example/auth/UserService.java /workspace/src/main/java/com/example/auth/UserService.java
COPY src/main/java/com/example/auth/AuthService.java /workspace/src/main/java/com/example/auth/AuthService.java
COPY src/main/java/com/example/auth/PermissionService.java /workspace/src/main/java/com/example/auth/PermissionService.java
COPY src/main/java/com/example/auth/AuthApplication.java /workspace/src/main/java/com/example/auth/AuthApplication.java

# Copy log file
COPY data/startup.log /workspace/logs/startup.log

WORKDIR /workspace