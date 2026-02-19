FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /src/com/example/app /src/com/example/utils /src/com/example/data /src/com/example/network /src/com/example/ui /src/com/example/config /src/com/example/deprecated /src/com/example/unused /src/com/example/test

COPY data/mapping.txt /data/mapping.txt
COPY src/com/example/app/MainActivity.java /src/com/example/app/MainActivity.java
COPY src/com/example/app/LoginActivity.java /src/com/example/app/LoginActivity.java
COPY src/com/example/utils/StringHelper.java /src/com/example/utils/StringHelper.java
COPY src/com/example/utils/Logger.java /src/com/example/utils/Logger.java
COPY src/com/example/data/UserModel.java /src/com/example/data/UserModel.java
COPY src/com/example/data/DatabaseHelper.java /src/com/example/data/DatabaseHelper.java
COPY src/com/example/network/ApiClient.java /src/com/example/network/ApiClient.java
COPY src/com/example/network/NetworkUtils.java /src/com/example/network/NetworkUtils.java
COPY src/com/example/ui/CustomButton.java /src/com/example/ui/CustomButton.java
COPY src/com/example/ui/DialogManager.java /src/com/example/ui/DialogManager.java
COPY src/com/example/config/AppConfig.java /src/com/example/config/AppConfig.java
COPY src/com/example/deprecated/OldHelper.java /src/com/example/deprecated/OldHelper.java
COPY src/com/example/unused/LegacyCode.java /src/com/example/unused/LegacyCode.java
COPY src/com/example/test/TestHelper.java /src/com/example/test/TestHelper.java

WORKDIR /workspace