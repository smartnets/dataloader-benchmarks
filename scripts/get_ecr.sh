#!/usr/bin/env bash
# get-ecr.sh

set -o allexport
source .env
set +o allexport

echo "y" | docker image prune
$(aws ecr get-login --no-include-email --region us-east-1)
docker pull "$DOCKER_NAME" 