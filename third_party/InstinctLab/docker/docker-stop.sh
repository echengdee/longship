#!/bin/bash

set -e
set -u

profile="instinctlab"
if [ $# -eq 1 ]
then
    profile=$1
fi

docker compose --file docker-compose.yaml --env-file .env.base down $profile
