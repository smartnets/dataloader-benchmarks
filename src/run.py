from src.dataloaders.cifar10.index import CIFAR10Loaders
from src.dataloaders.coco.index import CocoLoaders
from src.dataloaders.random.index import RandomLoaders
from src.models.index import MODEL_INDEX
import torch.multiprocessing as mp
from pprint import pprint
import time

from src.utils.distributed import setup, cleanup, get_open_port
import torch.distributed as dist

from src.config import settings as st
from src.train import setup as train_setup
from src.utils.general import should_run_test
from src.utils.time import get_current_timestamp
from src.utils.persist import is_s3_up, persist_results
from src.utils.general import config_to_bool

from src.utils.experiments import record_run, is_run_completed
import sys


import warnings

warnings.filterwarnings("ignore")

LOADER_SWITCHER = {
    "cifar10": CIFAR10Loaders,
    "coco": CocoLoaders,
    "random": RandomLoaders,
}


def main(rank, world_size, run_id, port):
    # setup the process groups

    record_run("new")

    distributed = config_to_bool(st.distributed)
    filtering = config_to_bool(st.filtering)

    if distributed:
        setup(rank, world_size, port)

    outputs = [None for _ in range(world_size)]

    loader = LOADER_SWITCHER[st.dataset][st.library]
    model = MODEL_INDEX[st.dataset]

    model, loaders, params, metric_logger = train_setup(model, loader, rank, run_id)

    train_loader = loaders["train"]
    for epoch in range(st.num_epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        if (
            distributed
            and hasattr(train_loader, "sampler")
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)
        model.train(train_loader, epoch, metric_logger, **params)

        if not filtering and int(st.cutoff) < 0:
            model.evaluate(loaders["val"], epoch, "val", metric_logger, **params)

    if should_run_test():
        if not filtering and int(st.cutoff) < 0:
            model.evaluate(loaders["test"], 0, "test", metric_logger, **params)

    metric_logger.end_side_collectors()
    res_dict = metric_logger.get_results_dict()
    if distributed:
        dist.all_gather_object(outputs, res_dict)
    else:
        outputs[rank] = res_dict

    if rank == 0:
        metric_logger.update_final_dicts(outputs)
        metric_logger.persist_metrics()

        if is_s3_up():
            persist_results(metric_logger.path)

    if distributed:
        cleanup()

    record_run("completed")


if __name__ == "__main__":

    id_ = get_current_timestamp()
    open_port = get_open_port()

    if is_run_completed():
        sys.exit()

    start = time.perf_counter()
    if config_to_bool(st.distributed):
        mp.spawn(main, args=(st.world_size, id_, open_port), nprocs=st.world_size)
    else:
        main(0, st.world_size, id_, open_port)

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f}")
