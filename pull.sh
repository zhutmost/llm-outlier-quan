#! /bin/sh

export SSH_USERNAME=yujiema
export SSH_IP=10.155.102.33
export PROJECT_NAME=oacim-camera

if [ $# -eq 0 ] ; then
	set -x
    scp -P 22 -r ${SSH_USERNAME}@${SSH_IP}:${PROJECT_NAME}/* .
else
    set -x
	scp -P 22 -r ${SSH_USERNAME}@${SSH_IP}:${PROJECT_NAME}/$1 ./$1
fi
