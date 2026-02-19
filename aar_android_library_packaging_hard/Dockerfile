FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Android SDK and required dependencies
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    wget \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk android-sdk zip unzip && \
    rm -rf /var/lib/apt/lists/*

# Set up Android SDK
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH="${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/33.0.0"

RUN mkdir -p ${ANDROID_HOME}/cmdline-tools && \
    cd ${ANDROID_HOME}/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip -q commandlinetools-linux-9477386_latest.zip && \
    mv cmdline-tools latest && \
    rm commandlinetools-linux-9477386_latest.zip

# Accept licenses and install Android SDK components
RUN yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses && \
    ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-28" "build-tools;33.0.0"

# Create workspace directory structure
RUN mkdir -p /workspace/java-library/src/com/example/library

# Copy generated Java source files to the container
COPY java-library/src/com/example/library/LogHelper.java /workspace/java-library/src/com/example/library/LogHelper.java
COPY java-library/src/com/example/library/StringUtils.java /workspace/java-library/src/com/example/library/StringUtils.java
COPY java-library/src/com/example/library/ContextHelper.java /workspace/java-library/src/com/example/library/ContextHelper.java
COPY java-library/src/com/example/library/NetworkUtils.java /workspace/java-library/src/com/example/library/NetworkUtils.java
COPY java-library/src/com/example/library/PreferenceHelper.java /workspace/java-library/src/com/example/library/PreferenceHelper.java

WORKDIR /workspace