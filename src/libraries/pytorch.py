from torch.utils.data import SubsetRandomSampler

def filter_by_class(active_classes, dataset):
    filtered_indices = [i for i, j in enumerate(dataset) if j[1] in active_classes]
    return SubsetRandomSampler(filtered_indices)


def filter_by_multi_class(active_classes, dataset):
    sampler = []
    ac = set(active_classes)
    for i, sample in enumerate(dataset):
        classes = set(x.item() for x in sample[1]["labels"])
        if len(classes.intersection(ac)) > 0:
            sampler.append(i)
    return sampler
