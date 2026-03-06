#!/bin/bash

cd /opt/jniapp/

echo "Running Java application with native library..."
echo "Library path: /usr/lib"

java -Djava.library.path=/usr/lib NativeApp