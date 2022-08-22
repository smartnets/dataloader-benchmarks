#!/usr/bin/env bash
python src/utils/setup_s3cmd.py
python src/datasets/prepare.py --library pytorch
python src/datasets/prepare.py --library ffcv
python src/datasets/prepare.py --library hub
python src/datasets/prepare.py --library hub --remote
python src/datasets/prepare.py --library hub3
python src/datasets/prepare.py --library hub3 --remote
python src/datasets/prepare.py --library webdataset
python src/datasets/prepare.py --library webdataset --remote
python src/datasets/prepare.py --library torchdata
python src/datasets/prepare.py --library squirrel
