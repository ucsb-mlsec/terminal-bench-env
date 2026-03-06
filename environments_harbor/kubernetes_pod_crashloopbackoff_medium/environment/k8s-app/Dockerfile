FROM python:3.9-slim

LABEL maintainer="devops@company.com"
LABEL version="1.0.0"

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]