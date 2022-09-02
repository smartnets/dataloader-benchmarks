#!/usr/bin/env bash

set -o allexport
source .env
set +o allexport

docker run \
     --rm \
     -it \
     --gpus device=all \
     --cap-add sys_ptrace \
     --privileged \
     --env-file .env \
     --shm-size 8G \
     -v benchmark-datasets:/home/worker/workspace/datasets \
     -v benchmark-plots:/home/worker/workspace/plots \
     -v benchmark-results:/home/worker/workspace/results \
     "$DOCKER_NAME" \
     "$@"
