FROM python:3.9-slim

WORKDIR /app

# TODO: Build times are really slow lately, need to investigate
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make

# Copy configuration files
COPY config/*.yml /app/config/

LABEL build_date="2024-01-15" version="1.0.0"

# Install Python packages - why is this always rebuilding?
RUN apt-get install -y curl wget

COPY requirements.txt /app/requirements.txt

# Copy additional source files
COPY scripts/*.sh /app/scripts/

RUN pip install --no-cache-dir -r requirements.txt

LABEL maintainer="dev-team@company.com"

# Install additional build tools
RUN apt-get install -y git

# Copy more config files
COPY *.json /app/

# Performance note: builds taking 15-20 minutes now, used to be 2-3 minutes

CMD ["python", "worker.py"]