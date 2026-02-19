FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Java and Maven for mutation testing
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk maven && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Create workspace directory structure
RUN mkdir -p /workspace/validator/src/main/java/com/example/validator && \
    mkdir -p /workspace/validator/src/test/java/com/example/validator

# Copy project files
COPY validator/pom.xml /workspace/validator/pom.xml
COPY validator/src/main/java/com/example/validator/StringValidator.java /workspace/validator/src/main/java/com/example/validator/StringValidator.java
COPY validator/src/test/java/com/example/validator/StringValidatorTest.java /workspace/validator/src/test/java/com/example/validator/StringValidatorTest.java

# Set working directory
WORKDIR /workspace