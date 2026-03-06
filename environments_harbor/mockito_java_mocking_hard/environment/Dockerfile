FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Maven and Java Development Kit
RUN apt-get update && \
    apt-get install -y maven openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create necessary directories
RUN mkdir -p /home/user/payment-app/src/main/java/com/example/payment \
             /home/user/payment-app/src/test/java/com/example/payment

# Copy pom.xml
COPY payment-app/pom.xml /home/user/payment-app/pom.xml

# Copy main Java files
COPY payment-app/src/main/java/com/example/payment/GatewayClient.java /home/user/payment-app/src/main/java/com/example/payment/GatewayClient.java
COPY payment-app/src/main/java/com/example/payment/TransactionLogger.java /home/user/payment-app/src/main/java/com/example/payment/TransactionLogger.java
COPY payment-app/src/main/java/com/example/payment/PaymentRequest.java /home/user/payment-app/src/main/java/com/example/payment/PaymentRequest.java
COPY payment-app/src/main/java/com/example/payment/PaymentResponse.java /home/user/payment-app/src/main/java/com/example/payment/PaymentResponse.java
COPY payment-app/src/main/java/com/example/payment/GatewayException.java /home/user/payment-app/src/main/java/com/example/payment/GatewayException.java
COPY payment-app/src/main/java/com/example/payment/PaymentService.java /home/user/payment-app/src/main/java/com/example/payment/PaymentService.java

WORKDIR /home/user/payment-app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
