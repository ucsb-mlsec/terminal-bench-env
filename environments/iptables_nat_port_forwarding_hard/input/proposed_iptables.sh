#!/bin/bash

# Emergency iptables configuration for gateway server
# Written during incident - needs review before production deployment

# Flush existing rules
iptables -F
iptables -t nat -F
iptables -X

# Set default policies
iptables -P INPUT ACCEPT
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# NAT Configuration
# POSTROUTING for internal network - allow outbound
iptables -t nat -A POSTROUTING -o eth0 -s 192.168.100.0/24 -j MASQUERADE

# Port Forwarding Rules - PREROUTING DNAT
# Web server - HTTP
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.100.10:80

# Web server - HTTPS (missing - forgotten during incident)

# Database - PostgreSQL (wrong IP address)
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 5432 -j DNAT --to-destination 192.168.100.15:5432

# API server - HTTP (wrong port)
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 8080 -j DNAT --to-destination 192.168.100.30:8000

# API server - HTTPS (wrong protocol)
iptables -t nat -A PREROUTING -i eth0 -p udp --dport 8443 -j DNAT --to-destination 192.168.100.30:8443

# Forward Chain Rules
# Allow all forwarding (temporary - should be more restrictive)
iptables -A FORWARD -j ACCEPT

# Specific forward rules for services
# Web server HTTP
iptables -A FORWARD -p tcp -d 192.168.100.10 --dport 80 -j ACCEPT

# Database (wrong destination IP matching DNAT error)
iptables -A FORWARD -p tcp -d 192.168.100.15 --dport 5432 -j ACCEPT

# API server HTTP (wrong port matching DNAT error)
iptables -A FORWARD -p tcp -d 192.168.100.30 --dport 8000 -j ACCEPT

# API server HTTPS (wrong protocol matching DNAT error)
iptables -A FORWARD -p udp -d 192.168.100.30 --dport 8443 -j ACCEPT

# Note: Need to enable IP forwarding in sysctl
# echo 1 > /proc/sys/net/ipv4/ip_forward