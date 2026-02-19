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
    apt-get install -y gcc build-essential binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/obj /workspace/lib

COPY obj/add.o /workspace/obj/add.o
COPY obj/subtract.o /workspace/obj/subtract.o
COPY obj/multiply.o /workspace/obj/multiply.o
COPY obj/divide.o /workspace/obj/divide.o
COPY lib/libcalc.a /workspace/lib/libcalc.a
COPY test_calc.c /workspace/test_calc.c

WORKDIR /workspace