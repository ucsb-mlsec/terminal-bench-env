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

RUN mkdir -p /workspace/library-app/src/main/java/com/library \
    /workspace/library-app/src/main/resources \
    /workspace

COPY workspace/library-app/pom.xml /workspace/library-app/pom.xml
COPY workspace/library-app/src/main/java/com/library/Book.java /workspace/library-app/src/main/java/com/library/Book.java
COPY workspace/library-app/src/main/java/com/library/Review.java /workspace/library-app/src/main/java/com/library/Review.java
COPY workspace/library-app/src/main/java/com/library/Main.java /workspace/library-app/src/main/java/com/library/Main.java
COPY workspace/library-app/src/main/java/com/library/HibernateUtil.java /workspace/library-app/src/main/java/com/library/HibernateUtil.java
COPY workspace/library-app/src/main/java/com/library/BookRepository.java /workspace/library-app/src/main/java/com/library/BookRepository.java
COPY workspace/library-app/src/main/resources/hibernate.cfg.xml /workspace/library-app/src/main/resources/hibernate.cfg.xml
COPY workspace/library-app/src/main/resources/init.sql /workspace/library-app/src/main/resources/init.sql

WORKDIR /workspace