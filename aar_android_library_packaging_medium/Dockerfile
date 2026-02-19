FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk wget unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/android-sdk/cmdline-tools && \
    cd /opt/android-sdk/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip -q commandlinetools-linux-9477386_latest.zip && \
    mv cmdline-tools latest && \
    rm commandlinetools-linux-9477386_latest.zip && \
    yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses && \
    ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0"

RUN mkdir -p /workspace/android-lib/app/src/main/java/com/example/androidlib && \
    mkdir -p /workspace/android-lib/app/src/main/res/values && \
    mkdir -p /workspace/android-lib/gradle/wrapper

COPY android-lib/build.gradle /workspace/android-lib/build.gradle
COPY android-lib/settings.gradle /workspace/android-lib/settings.gradle
COPY android-lib/app/build.gradle /workspace/android-lib/app/build.gradle
COPY android-lib/app/src/main/AndroidManifest.xml /workspace/android-lib/app/src/main/AndroidManifest.xml
COPY android-lib/app/src/main/java/com/example/androidlib/StringUtils.java /workspace/android-lib/app/src/main/java/com/example/androidlib/StringUtils.java
COPY android-lib/app/src/main/java/com/example/androidlib/MathUtils.java /workspace/android-lib/app/src/main/java/com/example/androidlib/MathUtils.java
COPY android-lib/app/src/main/java/com/example/androidlib/DateHelper.kt /workspace/android-lib/app/src/main/java/com/example/androidlib/DateHelper.kt
COPY android-lib/gradle.properties /workspace/android-lib/gradle.properties
COPY android-lib/gradlew /workspace/android-lib/gradlew
COPY android-lib/gradlew.bat /workspace/android-lib/gradlew.bat
COPY android-lib/gradle/wrapper/gradle-wrapper.properties /workspace/android-lib/gradle/wrapper/gradle-wrapper.properties
COPY android-lib/gradle/wrapper/gradle-wrapper.jar /workspace/android-lib/gradle/wrapper/gradle-wrapper.jar
COPY android-lib/app/src/main/res/values/strings.xml /workspace/android-lib/app/src/main/res/values/strings.xml
COPY android-lib/local.properties /workspace/android-lib/local.properties

RUN chmod +x /workspace/android-lib/gradlew

WORKDIR /workspace/android-lib