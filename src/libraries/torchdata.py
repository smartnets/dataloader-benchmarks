from torchdata.datapipes.map import SequenceWrapper
from torchdata.datapipes.iter import IterableWrapper


def build_dataset(
    data,
    distributed: bool = False,
):
    ds = IterableWrapper(data)

    # Using MapDataPipe class
    # ds = SequenceWrapper(random)
    if distributed:
        ds = ds.sharding_filter()
    return ds


def filter_by_class(active_classes, dataset):
    dataset = dataset.filter(filter_fn=lambda x: x[1] in active_classes)
    return dataset


def intersection_nonempty(list_a, list_b):
    set_a = set(x.item() for x in list_a)
    set_b = set(list_b)
    return len(set_a.intersection(set_b)) > 0


def _filter_by_multiclass(x, y):
    return intersection_nonempty(x[1]["labels"], y)


def filter_by_multiclass(active_classes, dataset):
    dataset = dataset.filter(
        filter_fn=lambda x: _filter_by_multiclass(x, active_classes)
    )
    return dataset
