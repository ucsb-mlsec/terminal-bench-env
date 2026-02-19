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
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create workspace directory
RUN mkdir -p /workspace/notification-app/src/main/java/com/notification /workspace/notification-app/src/main/resources

# Set working directory
WORKDIR /workspace

# Copy project files
COPY notification-app/pom.xml /workspace/notification-app/pom.xml
COPY notification-app/src/main/resources/application.properties /workspace/notification-app/src/main/resources/application.properties
COPY notification-app/src/main/java/com/notification/NotificationApplication.java /workspace/notification-app/src/main/java/com/notification/NotificationApplication.java
COPY notification-app/src/main/java/com/notification/NotificationService.java /workspace/notification-app/src/main/java/com/notification/NotificationService.java
COPY notification-app/src/main/java/com/notification/UserPreferenceService.java /workspace/notification-app/src/main/java/com/notification/UserPreferenceService.java
COPY notification-app/src/main/java/com/notification/NotificationDeliveryService.java /workspace/notification-app/src/main/java/com/notification/NotificationDeliveryService.java