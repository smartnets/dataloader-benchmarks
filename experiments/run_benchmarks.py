import argparse
import subprocess
import time
from itertools import product

from experiments.setup import configure_env
from experiments.configuration import EXPERIMENTS
from src.utils.setup_s3cmd import setup_s3cmd


def skip_experiment(bs, nw, lb):
    if nw >= bs:
        return True
    if lb == "hub3" and nw >= 2:
        return True
    return False


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        help="Dataset to run the experiments with: cifar10|random|coco [default: random]",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        default=False,
        dest="multi_gpu",
        help="Use this flag to enable multi-gpu usage [default: False]",
    )
    parser.add_argument(
        "--filtering",
        action="store_true",
        default=False,
        dest="filtering",
        help="Use this flag to enable filtering [default: False]",
    )
    args = parser.parse_args()

    setup_s3cmd()

    gpu_mode = "multi-gpu" if args.multi_gpu else "single-gpu"

    if args.filtering:
        gpu_mode = "filtering"

    experiment_dict = EXPERIMENTS[args.dataset]
    BS = experiment_dict["batch_size"]
    NW = experiment_dict["workers"]
    LB = experiment_dict["libraries"][gpu_mode]
    CO = experiment_dict["cutoff"]
    RM = experiment_dict["is_cutoff_run_model"]
    FC = experiment_dict["filtering_classes"]
    REPS = experiment_dict["reps"]

    for rep in range(REPS):
        for batch_size, num_workers, library, run_model in product(BS, NW, LB, RM):

            if skip_experiment(batch_size, num_workers, library):
                continue

            configure_env(
                args.dataset,
                library,
                num_workers,
                CO,
                batch_size,
                run_model,
                args.multi_gpu,
                args.filtering,
                FC,
            )
            ARGS = ["python", "src/run.py"]
            pid = subprocess.Popen(ARGS)
            pid.wait()

            time.sleep(1)
