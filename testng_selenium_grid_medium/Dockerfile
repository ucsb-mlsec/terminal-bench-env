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
    apt-get install -y openjdk-17-jdk maven && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN mkdir -p /workspace/grid_tests/src/main/java/com/qa/automation/grid && \
    mkdir -p /workspace/grid_tests/src/test/java/com/qa/automation/tests

COPY grid_tests/pom.xml /workspace/grid_tests/pom.xml
COPY grid_tests/testng.xml /workspace/grid_tests/testng.xml
COPY grid_tests/src/main/java/com/qa/automation/grid/MockGridCoordinator.java /workspace/grid_tests/src/main/java/com/qa/automation/grid/MockGridCoordinator.java
COPY grid_tests/src/main/java/com/qa/automation/grid/MockWebDriver.java /workspace/grid_tests/src/main/java/com/qa/automation/grid/MockWebDriver.java
COPY grid_tests/src/main/java/com/qa/automation/grid/MockWebElement.java /workspace/grid_tests/src/main/java/com/qa/automation/grid/MockWebElement.java
COPY grid_tests/src/test/java/com/qa/automation/tests/LoginTest.java /workspace/grid_tests/src/test/java/com/qa/automation/tests/LoginTest.java
COPY grid_tests/src/test/java/com/qa/automation/tests/FormTest.java /workspace/grid_tests/src/test/java/com/qa/automation/tests/FormTest.java

WORKDIR /workspace