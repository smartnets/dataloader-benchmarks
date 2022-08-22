#!/usr/bin/env bash
python src/utils/setup_s3cmd.py
python src/datasets/prepare.py --dataset random --library pytorch # needs to be the first one
python src/datasets/prepare.py --dataset random --library ffcv
python src/datasets/prepare.py --dataset random --library hub
python src/datasets/prepare.py --dataset random --library hub --remote
python src/datasets/prepare.py --dataset random --library hub3
python src/datasets/prepare.py --dataset random --library hub3 --remote
python src/datasets/prepare.py --dataset random --library squirrel
python src/datasets/prepare.py --dataset random --library torchdata
python src/datasets/prepare.py --dataset random --library webdataset
python src/datasets/prepare.py --dataset random --library webdataset --remote