from functools import partial
import webdataset as wds
import torch
import torch.distributed as dist
from pathlib import Path
from src.datasets.transformations.identity import Identity
from src.utils.distributed import get_worker_info
from functools import partial


def filter_by_class(active_labels, sample):
    return sample[1] in active_labels


def intersection_nonempty(list_a, list_b):
    set_a = set(x.item() for x in list_a)
    set_b = set(list_b)
    return len(set_a.intersection(set_b)) > 0


def filter_by_multiclass(active_classes, dataset):
    dataset = dataset.filter(
        filter_fn=lambda x: intersection_nonempty(x[1]["labels"], active_classes)
    )
    return dataset


def my_split_shards(distributed, urls):
    total_workers, global_worker_id = get_worker_info()
    for i, s in enumerate(urls):
        if i % total_workers == global_worker_id:
            yield s


def build_dataset(
    path: Path,
    distributed: bool = False,
    transformations=None,
    filtering: bool = False,
    filtering_classes: list = [],
    batch_size: int = 4,
    shuffle: bool = False,
    dataset_kind: str = "classification",
):

    if dataset_kind == "coco":
        filter = partial(filter_by_multiclass, filtering_classes)
    else:
        filter = partial(filter_by_class, filtering_classes)

    split = partial(my_split_shards, distributed)

    ds = wds.WebDataset(path, nodesplitter=split).decode("pil")

    if dataset_kind == "coco":
        ds = ds.to_tuple("image.jpg", "labels.npy", "boxes.npy")
    else:
        ds = ds.to_tuple("jpg", "cls")

    if filtering:
        ds = ds.select(filter)

    if dataset_kind == "coco":
        ds = ds.map(transformations)
    else:
        ds = ds.map_tuple(transformations, Identity())

    ds = ds.batched(batch_size, partial=(not distributed))
    return ds


def build_loader(
    dataset, size: int, batch_size: int, num_workers: int, distributed: bool = False
):

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    number_of_batches = size // batch_size
    if distributed:
        loader = loader.repeat(2).slice(number_of_batches)
        loader.length = number_of_batches
    return loader
