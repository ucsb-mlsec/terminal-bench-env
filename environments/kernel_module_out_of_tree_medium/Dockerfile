FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/developer/kernel_modules/module_a \
    /home/developer/kernel_modules/module_b \
    /home/developer/kernel_modules/module_c

COPY module_a/module_a_driver.c /home/developer/kernel_modules/module_a/module_a_driver.c
COPY module_a/Makefile /home/developer/kernel_modules/module_a/Makefile
COPY module_b/network_filter.c /home/developer/kernel_modules/module_b/network_filter.c
COPY module_b/Makefile /home/developer/kernel_modules/module_b/Makefile
COPY module_c/usb_device.c /home/developer/kernel_modules/module_c/usb_device.c
COPY module_c/Makefile /home/developer/kernel_modules/module_c/Makefile

WORKDIR /home/developer