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
    apt-get install -y gfortran make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/fortran_project/src

COPY fortran_project/src/module_constants.f90 /home/fortran_project/src/
COPY fortran_project/src/module_math.f90 /home/fortran_project/src/
COPY fortran_project/src/module_compute.f90 /home/fortran_project/src/
COPY fortran_project/src/main.f90 /home/fortran_project/src/
COPY fortran_project/Makefile /home/fortran_project/

WORKDIR /home/fortran_project

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
