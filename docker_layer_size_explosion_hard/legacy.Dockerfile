FROM python:3.9

LABEL maintainer="devops@example.com"

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

EXPOSE 8080

WORKDIR /app

RUN apt-get update

RUN apt-get install -y curl

RUN apt-get install -y wget git

RUN apt-get install -y build-essential gcc

RUN apt-get install -y libpq-dev

RUN apt-get install -y netcat

RUN pip install flask

RUN pip install requests

RUN pip install numpy pandas

RUN pip install psycopg2-binary

RUN pip install gunicorn

RUN curl -O https://downloads.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz && tar -xzf apache-maven-3.8.6-bin.tar.gz

RUN wget https://github.com/prometheus/node_exporter/releases/download/v1.3.1/node_exporter-1.3.1.linux-amd64.tar.gz && tar -xzf node_exporter-1.3.1.linux-amd64.tar.gz

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY app.py /app/app.py

COPY config.py /app/config.py

RUN useradd -m -u 1000 appuser

RUN chown -R appuser:appuser /app

USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]