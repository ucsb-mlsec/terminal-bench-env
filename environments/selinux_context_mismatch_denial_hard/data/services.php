ISSUE: SELinux context preventing httpd from reading files in /opt/webapp/html
FIX_COMMAND: semanage fcontext -a -t httpd_sys_content_t "/opt/webapp/html(/.*)?" && restorecon -Rv /opt/webapp/html
PERSISTENCE: yes