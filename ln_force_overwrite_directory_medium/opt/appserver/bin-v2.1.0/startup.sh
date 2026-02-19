#!/bin/bash

# Application Startup Script v2.1.0
echo 'Starting application server version 2.1.0'
export APP_HOME=/opt/appserver
export JAVA_OPTS='-Xmx1024m -Xms512m'
java $JAVA_OPTS -jar $APP_HOME/lib/commons.jar