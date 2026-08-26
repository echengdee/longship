#!/bin/bash

set -e
set -u

profile="instinctlab"
if [ $# -eq 1 ]
then
    profile=$1
fi

docker exec --interactive --tty $profile bash
