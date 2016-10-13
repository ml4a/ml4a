#!/usr/bin/env bash

#adapted from https://github.com/kylemcdonald/ml-notebook/blob/master/run.sh

CONTAINER="ml4a-guides"
IMAGE="genekogan/$CONTAINER"
IMAGE_FILE="$CONTAINER.tar"
VM="default"
JUPYTER_PORT="8888"

source start-docker.sh

if ! ( docker images | grep "$IMAGE" &>/dev/null ) ; then
	if [ -e $IMAGE_FILE ]; then
		echo "The image will be loaded from $IMAGE_FILE (first time only, ~1 minute)."
		docker load < $IMAGE_FILE
	else
		echo "The image will be downloaded from docker (first time only)."
	fi
fi

HOST_IP=`docker-machine ip $VM`

# this might be better as an ssh followed by a deletion on exit
if ( docker stats --no-stream=true $CONTAINER &>/dev/null ) ; then
	echo "The container is already running, stopping it..."
	docker stop $CONTAINER &>/dev/null
	if ( docker stats --no-stream=true $CONTAINER &>/dev/null ) ; then
		echo "The container still exists, removing it..."
		docker rm $CONTAINER &>/dev/null
	fi
fi

openavailable() {
	# echo "Opening http://$1:$2/ once it is available..."
	# wait for nc to connect to the port
	# todo: windows?
	until nc -vz "$1" "$2" &>/dev/null; do
		sleep 1
	done
	# todo: windows and linux?
	open "http://$1:$2/"
}

openavailable $HOST_IP $JUPYTER_PORT &

DIR=`pwd`
docker run -ti \
	--rm=true \
	--name="$CONTAINER" \
	--publish="$JUPYTER_PORT:$JUPYTER_PORT" \
	--env "HOST_IP=$HOST_IP" \
	--workdir="/root" \
	--volume="$DIR:/root" \
	$IMAGE \
	/bin/bash -c " \
		sudo ln /dev/null /dev/raw1394 ; \
		jupyter notebook --ip='*' --no-browser > jupyter.log 2>&1 & \
		echo 'Jupyter is at http://$HOST_IP:$JUPYTER_PORT/ and writing to jupyter.log' ; \
		bash"
