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
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/employee-app/src/main/java/com/company/model && \
    mkdir -p /workspace/employee-app/src/main/java/com/company/util && \
    mkdir -p /workspace/employee-app/src/main/resources

COPY workspace/employee-app/pom.xml /workspace/employee-app/pom.xml
COPY workspace/employee-app/src/main/java/com/company/model/Department.java /workspace/employee-app/src/main/java/com/company/model/Department.java
COPY workspace/employee-app/src/main/java/com/company/model/Employee.java /workspace/employee-app/src/main/java/com/company/model/Employee.java
COPY workspace/employee-app/src/main/java/com/company/util/HibernateUtil.java /workspace/employee-app/src/main/java/com/company/util/HibernateUtil.java
COPY workspace/employee-app/src/main/java/com/company/Main.java /workspace/employee-app/src/main/java/com/company/Main.java
COPY workspace/employee-app/src/main/resources/hibernate.cfg.xml /workspace/employee-app/src/main/resources/hibernate.cfg.xml
COPY workspace/employee-app/src/main/resources/import.sql /workspace/employee-app/src/main/resources/import.sql

WORKDIR /workspace