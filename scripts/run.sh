#!/usr/bin/env bash
# run.sh
docker run \
     --rm \
     -it \
     --gpus device=all \
     --cap-add sys_ptrace \
     --privileged \
     --env-file .env \
     -v benchmarks-datasets:/home/worker/workspace/datasets \
     -v benchmarks-results:/home/worker/workspace/results \
     ml-benchmarks:2.0  \
     "$@"
