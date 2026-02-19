FROM python:3.11

WORKDIR /app

# FIXME: Cache inefficiency here
# Copying all application code before installing dependencies
COPY . /app

# Installing Python dependencies after code copy
# This invalidates cache on every code change
RUN pip install --no-cache-dir -r requirements.txt

# FIXME: No BuildKit cache mount for pip
# Every build downloads packages again

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the application
CMD ["python", "main.py"]