#!/bin/bash

# QEMU Virtual Machine Launch Script
# Production deployment configuration for web server VM

qemu-system-x86_64 \
  -name "production-web-vm" \
  -machine type=q35,accel=kvm \
  -cpu host \
  -m 4096 \
  -smp cores=8,threads=1,sockets=1 \
  -boot order=c \
  -drive file=/var/lib/qemu/images/web-server.qcow2,if=virtio,format=qcow2 \
  -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
  -device virtio-net-pci,netdev=net0,mac=52:54:00:12:34:56 \
  -display vnc=:0,password=on \
  -monitor unix:/var/run/qemu-web-vm.sock,server,nowait \
  -rtc base=utc,clock=host \
  -device virtio-balloon-pci \
  -device virtio-rng-pci \
  -serial file:/var/log/qemu/web-vm-serial.log \
  -daemonize \
  -pidfile /var/run/qemu-web-vm.pid

# Configuration Summary:
# - Memory: 4096 MB
# - CPUs: 8 cores
# - Network: TAP interface (bridged networking)
# - Display: VNC on port 5900
# - Disk: virtio driver for optimal performance
# - Machine: Q35 chipset with KVM acceleration