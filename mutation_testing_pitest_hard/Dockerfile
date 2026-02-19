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
    apt-get install -y maven openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN mkdir -p /workspace/game-scoring/src/main/java/com/game && \
    mkdir -p /workspace/game-scoring/src/test/java/com/game

COPY workspace/game-scoring/pom.xml /workspace/game-scoring/pom.xml
COPY workspace/game-scoring/src/main/java/com/game/ScoreCalculator.java /workspace/game-scoring/src/main/java/com/game/ScoreCalculator.java
COPY workspace/game-scoring/src/main/java/com/game/PlayerRanking.java /workspace/game-scoring/src/main/java/com/game/PlayerRanking.java
COPY workspace/game-scoring/src/main/java/com/game/BonusMultiplier.java /workspace/game-scoring/src/main/java/com/game/BonusMultiplier.java
COPY workspace/game-scoring/src/test/java/com/game/ScoreCalculatorTest.java /workspace/game-scoring/src/test/java/com/game/ScoreCalculatorTest.java
COPY workspace/game-scoring/src/test/java/com/game/PlayerRankingTest.java /workspace/game-scoring/src/test/java/com/game/PlayerRankingTest.java
COPY workspace/game-scoring/src/test/java/com/game/BonusMultiplierTest.java /workspace/game-scoring/src/test/java/com/game/BonusMultiplierTest.java

WORKDIR /workspace