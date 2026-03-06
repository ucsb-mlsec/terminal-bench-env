FROM ubuntu:latest AS

LABEL maintainer="devops@example.com"
LABEL version="1.0"

RUN apt-get update && \
    apt-get install -y openjdk-11-jre && \
    echo "installing packages

WORKDIR /app

COPY app.jar

RUN chmod +x /app/app.jar

EXPOSE 8080 9090 TCP

ENV JAVA_OPTS="-Xmx512m -Xms256m"
ENV APP_PORT=8080

CMD ["java", "-jar" "app.jar"]