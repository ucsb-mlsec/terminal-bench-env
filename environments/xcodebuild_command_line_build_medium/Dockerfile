FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Android SDK dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gradle openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install Android SDK
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

RUN mkdir -p $ANDROID_HOME/cmdline-tools && \
    cd $ANDROID_HOME/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip commandlinetools-linux-9477386_latest.zip && \
    mv cmdline-tools latest && \
    rm commandlinetools-linux-9477386_latest.zip

# Accept licenses and install required SDK components
RUN yes | sdkmanager --licenses && \
    sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0"

# Create workspace directory structure
RUN mkdir -p /workspace/MobileApp/app/src/main/java/com/example/mobileapp \
    /workspace/MobileApp/app/src/main/res/layout \
    /workspace/MobileApp/app/src/main/res/values \
    /workspace/MobileApp/gradle/wrapper

# Copy all project files
COPY MobileApp/build.gradle /workspace/MobileApp/build.gradle
COPY MobileApp/app/build.gradle /workspace/MobileApp/app/build.gradle
COPY MobileApp/settings.gradle /workspace/MobileApp/settings.gradle
COPY MobileApp/gradle.properties /workspace/MobileApp/gradle.properties
COPY MobileApp/local.properties /workspace/MobileApp/local.properties
COPY MobileApp/app/src/main/AndroidManifest.xml /workspace/MobileApp/app/src/main/AndroidManifest.xml
COPY MobileApp/app/src/main/java/com/example/mobileapp/MainActivity.java /workspace/MobileApp/app/src/main/java/com/example/mobileapp/MainActivity.java
COPY MobileApp/app/src/main/res/layout/activity_main.xml /workspace/MobileApp/app/src/main/res/layout/activity_main.xml
COPY MobileApp/app/src/main/res/values/strings.xml /workspace/MobileApp/app/src/main/res/values/strings.xml
COPY MobileApp/gradle/wrapper/gradle-wrapper.properties /workspace/MobileApp/gradle/wrapper/gradle-wrapper.properties

WORKDIR /workspace