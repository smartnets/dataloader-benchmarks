#!/usr/bin/env bash
NAME="results/results_all_experiments.txt"
mkdir -p results
mkdir -p results/plots

function print_time {
    now=$(date)
    echo "$now: $@" >> "$NAME"
}

## Uncomment based on your experiments

print_time "prepare cifar" && ./experiments/prepare_cifar10.sh 
# print_time "prepare random" && ./experiments/prepare_random.sh
# print_time "prepare coco" && ./experiments/prepare_coco.sh

print_time "cifar10 def" && python experiments/run_benchmarks.py --dataset cifar10 --filename $NAME
# print_time "random def" && python experiments/run_benchmarks.py --dataset random --filename $NAME
# print_time "coco def" && python experiments/run_benchmarks.py --dataset coco --filename $NAME

# print_time "cifar10 filter" && python experiments/run_benchmarks.py --dataset cifar10 --filtering --filename $NAME
# print_time "random filter" && python experiments/run_benchmarks.py --dataset random --filtering --filename $NAME
# print_time "coco filter" && python experiments/run_benchmarks.py --dataset coco --filtering --filename $NAME

# print_time "cifar10 multi" && python experiments/run_benchmarks.py --dataset cifar10 --multi-gpu  --filename $NAME
# print_time "random multi" && python experiments/run_benchmarks.py --dataset random --multi-gpu  --filename $NAME
# print_time "coco multi" && python experiments/run_benchmarks.py --dataset coco --multi-gpu  --filename $NAME