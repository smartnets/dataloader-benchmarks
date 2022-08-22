#!/usr/bin/env bash
docker run \
  -p 10000:9000 \
  -p 10001:9001 \
  -e "MINIO_ROOT_USER=username" \
  -e "MINIO_ROOT_PASSWORD=password" \
  -v "data:/data" \
  quay.io/minio/minio server /data --console-address ":9001" &