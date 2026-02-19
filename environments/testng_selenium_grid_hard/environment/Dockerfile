FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /workspace/web-tests/src/test/java/com/example/tests

# Install Java, Maven, and dependencies for Selenium Grid
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    maven \
    wget \
    curl \
    gnupg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome and ChromeDriver for Selenium Grid
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jdk maven wget unzip && \
    pip3 install --no-cache-dir --break-system-packages selenium && \
    rm -rf /var/lib/apt/lists/*

# Download Selenium Server for Grid setup
RUN wget -q https://github.com/SeleniumHQ/selenium/releases/download/selenium-4.15.0/selenium-server-4.15.0.jar -O /opt/selenium-server.jar

# Copy all project files
COPY workspace/web-tests/pom.xml /workspace/web-tests/pom.xml
COPY workspace/web-tests/testng.xml /workspace/web-tests/testng.xml
COPY workspace/web-tests/src/test/java/com/example/tests/WebTest.java /workspace/web-tests/src/test/java/com/example/tests/WebTest.java
COPY workspace/web-tests/src/test/java/com/example/tests/LoginTest.java /workspace/web-tests/src/test/java/com/example/tests/LoginTest.java

WORKDIR /workspace/web-tests

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
