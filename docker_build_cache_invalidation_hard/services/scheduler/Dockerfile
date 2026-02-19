FROM python:3.9 as builder

ARG APP_VERSION=1.0.0
ARG BUILD_DATE
LABEL version=${APP_VERSION}

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-dev.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

RUN python -m compileall .

FROM python:3.9-slim

ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}

WORKDIR /app

COPY scheduler.py .
COPY utils.py .
COPY config.yaml .

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["python", "scheduler.py"]