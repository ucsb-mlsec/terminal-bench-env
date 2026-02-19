FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directory structure
RUN mkdir -p /forensics/tomcat/conf \
    /forensics/tomcat/webapps/ROOT/WEB-INF/classes \
    /forensics/tomcat/webapps/manager/WEB-INF \
    /forensics/tomcat/logs \
    /home/agent

# Copy forensic evidence files
COPY data/forensics/ajp_requests.log /forensics/ajp_requests.log
COPY data/forensics/tomcat/conf/server.xml /forensics/tomcat/conf/server.xml
COPY data/forensics/tomcat/conf/tomcat-users.xml /forensics/tomcat/conf/tomcat-users.xml
COPY data/forensics/tomcat/conf/context.xml /forensics/tomcat/conf/context.xml
COPY data/forensics/tomcat/conf/web.xml /forensics/tomcat/conf/web.xml
COPY data/forensics/tomcat/conf/catalina.properties /forensics/tomcat/conf/catalina.properties
COPY data/forensics/tomcat/webapps/ROOT/WEB-INF/classes/database.properties /forensics/tomcat/webapps/ROOT/WEB-INF/classes/database.properties
COPY data/forensics/tomcat/webapps/ROOT/WEB-INF/applicationContext.xml /forensics/tomcat/webapps/ROOT/WEB-INF/applicationContext.xml
COPY data/forensics/tomcat/webapps/manager/WEB-INF/web.xml /forensics/tomcat/webapps/manager/WEB-INF/web.xml
COPY data/forensics/tomcat/logs/catalina.out /forensics/tomcat/logs/catalina.out
COPY data/forensics/suspicious_files.txt /forensics/suspicious_files.txt

WORKDIR /home/agent