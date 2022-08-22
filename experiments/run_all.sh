#!/usr/bin/env bash

./experiments/prepare_cifar10.sh
./experiments/prepare_random.sh
./experiments/prepare_coco.sh

python experiments/run_benchmarks.py --dataset cifar10
python experiments/run_benchmarks.py --dataset cifar10 --multi-gpu
python experiments/run_benchmarks.py --dataset cifar10 --filtering

python experiments/run_benchmarks.py --dataset random
python experiments/run_benchmarks.py --dataset random --multi-gpu
python experiments/run_benchmarks.py --dataset random --filtering

python experiments/run_benchmarks.py --dataset coco
python experiments/run_benchmarks.py --dataset coco --multi-gpu
python experiments/run_benchmarks.py --dataset coco --filtering