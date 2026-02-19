FROM python:3.9 as base

ARG BUILD_DATE=unknown
ARG VERSION=1.0.0
LABEL build_date=${BUILD_DATE}

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    curl \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest flake8 black

COPY config/app.conf /etc/app/
COPY config/nginx.conf /etc/nginx/

ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

ARG GIT_COMMIT=unknown
LABEL git_commit=${GIT_COMMIT}

RUN useradd -m appuser && chown -R appuser:appuser /app

FROM base as production

COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

WORKDIR /app

USER appuser

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]