FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/lib/kubelet \
    && mkdir -p /etc/systemd/system/kubelet.service.d \
    && mkdir -p /var/log \
    && mkdir -p /etc/kubernetes \
    && mkdir -p /opt/cni/bin \
    && mkdir -p /etc/cni/net.d \
    && mkdir -p /root

COPY config/kubelet_config.yaml /var/lib/kubelet/config.yaml
COPY config/10-kubeadm.conf /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
COPY data/kubelet.log /var/log/kubelet.log
COPY config/kubelet.conf /etc/kubernetes/kubelet.conf
COPY data/cni_bin_listing.txt /opt/cni/bin/listing.txt
COPY data/cni_config_listing.txt /etc/cni/net.d/listing.txt

WORKDIR /root