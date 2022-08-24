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
     -v benchmarks-datasets:/home/worker/workspace/datasets \
     -v benchmarks-plots:/home/worker/workspace/plots \
     -v benchmarks-results:/home/worker/workspace/results \
     "$DOCKER_NAME" \
     "$@"
