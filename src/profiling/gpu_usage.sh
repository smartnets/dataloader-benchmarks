#!/usr/bin/env bash
OUTFILE="$1"
rm -f "$OUTFILE"

trap exit SIGINT;

while true;
do
    /usr/bin/nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory --format=csv | tail -n +2 >> "$OUTFILE" 
    sleep 0.5
done