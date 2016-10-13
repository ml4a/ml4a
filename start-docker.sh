#!/usr/bin/env bash

# this script is a simplified version of:
# /Applications/Docker/Docker Quickstart Terminal.app/Contents/Resources/Scripts/start.sh

VM=default
DOCKER_MACHINE=/usr/local/bin/docker-machine
VBOXMANAGE=/Applications/VirtualBox.app/Contents/MacOS/VBoxManage

unset DYLD_LIBRARY_PATH
unset LD_LIBRARY_PATH

if [ ! -f $DOCKER_MACHINE ] || [ ! -f $VBOXMANAGE ]; then
  echo "Either VirtualBox or Docker Machine are not installed. Please re-run the Toolbox Installer and try again."
  exit 1
fi

$VBOXMANAGE showvminfo $VM &> /dev/null
VM_EXISTS_CODE=$?

if [ $VM_EXISTS_CODE -eq 1 ]; then
  $DOCKER_MACHINE rm -f $VM &> /dev/null
  rm -rf ~/.docker/machine/machines/$VM
  $DOCKER_MACHINE create -d virtualbox --virtualbox-memory 2048 --virtualbox-disk-size 204800 $VM
fi

VM_STATUS=$($DOCKER_MACHINE status $VM 2>&1)
if [ "$VM_STATUS" != "Running" ]; then
  $DOCKER_MACHINE start $VM
  yes | $DOCKER_MACHINE regenerate-certs $VM
fi

eval $($DOCKER_MACHINE env $VM --shell=bash)
