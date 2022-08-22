#!/usr/bin/env bash

set -o allexport
source .env 
set +o allexport

docker build -f infrastructure/Dockerfile -t "$DOCKER_NAME" .

