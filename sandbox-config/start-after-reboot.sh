#!/bin/bash

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script as root or use sudo."
    exit 1
fi

# Set the ip address for the vboxnet0 interface
ifconfig vboxnet0 192.168.23.1/24 up

# Start cape sandbox in Docker
docker start cape
docker start file-uploader # Start the file uploader service

# Restart all the services
docker exec cape /bin/bash -c "systemctl restart cape"
docker exec cape /bin/bash -c "systemctl restart cape-web"
docker exec cape /bin/bash -c "systemctl restart cape-processor"
docker exec cape /bin/bash -c "systemctl restart cape-rooter"

