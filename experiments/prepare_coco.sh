#!/usr/bin/env bash
python ../src/datasets/prepare.py --dataset coco --library pytorch
python ../src/datasets/prepare.py --dataset coco --library webdataset
python ../src/datasets/prepare.py --dataset coco --library hub
python ../src/datasets/prepare.py --dataset coco --library hub --remote
python ../src/datasets/prepare.py --dataset coco --library hub3 --remote
python ../src/datasets/prepare.py --dataset coco --library hub3