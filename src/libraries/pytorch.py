def filter_by_class(active_classes, dataset):
    sampler = [i for i, j in enumerate(dataset) if j[1] in active_classes]
    return sampler


def filter_by_multi_class(active_classes, dataset):
    sampler = []
    ac = set(active_classes)
    for i, sample in enumerate(dataset):
        classes = set(x.item() for x in sample[1]["labels"])
        if len(classes.intersection(ac)) > 0:
            sampler.append(i)
    return sampler