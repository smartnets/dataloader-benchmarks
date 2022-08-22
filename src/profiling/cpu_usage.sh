#!/usr/bin/env bash

PID="$1"
OUTFILE="$2"

rm -f "$OUTFILE"

trap exit SIGINT SEGV;

while ps --pid $PID &>/dev/null;
do
    # Needs to be edited, as it results to error
    # depending on the screen size of the terminal running
    top -p $PID -b -H -n1 | awk 'NR > 7 {print $1","$9","$10}' | paste -s -d "|" >> "$OUTFILE"
    sleep 0.5
done
