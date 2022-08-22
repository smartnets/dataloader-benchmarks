import argparse
from src.datasets.cifar10.index import CIFAR10Datasets
from src.datasets.coco.index import CocoDatasets
from src.datasets.random.index import RandomDatasets
from pathlib import Path
from src.config import settings as st

default_data_dir = Path(st.local_data_dir)
remote_data_dir = Path(st.remote_data_dir)


DATASET_SWITCHER = {
    "cifar10": CIFAR10Datasets,
    "coco": CocoDatasets,
    "random": RandomDatasets
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Prepare Dataset for Benchmarks')
    parser.add_argument('--dataset', type=str, default="cifar10", metavar='dataset',
                        help='Dataset to work with. Options: cifar10|coco|random [default: cifar10]')
    parser.add_argument('--library', type=str, default="pytorch", metavar='library',
                        help='Library to load the dataset. Options: pytorch|ffcv|hub|webdataset|torchdata|squirrel|hub3 [default: pytorch]')
    parser.add_argument('--path', type=str, default=f"{default_data_dir}", metavar='path',
                        help=f'Path to store the data. Options: pytorch|ffcv [default: {default_data_dir}]')
    parser.add_argument('--remote', default=False, action="store_true",
                        help=f'If the dataset should be prepared remotely [default: False]')

    args = parser.parse_args()

    try:
        dataset = DATASET_SWITCHER[args.dataset][args.library]
    except IndexError:
        print(f"Either {args.dataset} or {args.library} is not valid")

    path = Path(args.path) / args.dataset / args.library
    print(path)
    remote_path = remote_data_dir / args.dataset / args.library

    modes = ["train", "val"]
    if args.dataset == "cifar10":
        modes.append("test")

    for mode in modes:
        if args.remote:
            print("Generating dataset remotely")
            dataset.generate_remotely(mode=mode)
        else:
            dataset.generate_locally(mode=mode)
