FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Maven and required tools
RUN apt-get update && \
    apt-get install -y maven && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/microservices-app/auth-service \
    /workspace/microservices-app/data-service \
    /workspace/microservices-app/api-gateway \
    /workspace/microservices-app/common-lib

# Copy all POM files to their correct locations
COPY microservices-app/pom.xml /workspace/microservices-app/pom.xml
COPY microservices-app/auth-service/pom.xml /workspace/microservices-app/auth-service/pom.xml
COPY microservices-app/data-service/pom.xml /workspace/microservices-app/data-service/pom.xml
COPY microservices-app/api-gateway/pom.xml /workspace/microservices-app/api-gateway/pom.xml
COPY microservices-app/common-lib/pom.xml /workspace/microservices-app/common-lib/pom.xml

# Set working directory
WORKDIR /workspace