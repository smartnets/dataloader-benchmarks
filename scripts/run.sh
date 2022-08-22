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
     -v benchmarks-datasets:/home/worker/workspace/datasets \
     -v benchmarks-results:/home/worker/workspace/results \
     "$DOCKER_NAME" \
     "$@"
