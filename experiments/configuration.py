EXPERIMENTS = {
    "random": {
        "batch_size": [32],
        "workers": [0, 1, 2],
        "libraries": {
            "single-gpu": [
                "ffcv",
                "hub",
                "hub-remote",
                "hub3",
                "hub3-remote",
                "pytorch",
                "squirrel",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
            "multi-gpu": [
                "ffcv",
                "pytorch",
                "hub3",
                "hub3-remote",
                "squirrel",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
            "filtering": [
                "hub",
                "hub-remote",
                "hub3",
                "hub3-remote",
                "pytorch",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
        },
        "is_cutoff_run_model": [True],
        "filtering_classes": ["0", "13"],
        "cutoff": 10,
    },
    "cifar10": {
        "batch_size": [16, 32],
        "workers": [0, 1, 2],
        "libraries": {
            "single-gpu": [
                "ffcv",
                "hub",
                "hub-remote",
                "hub3",
                "hub3-remote",
                "pytorch",
                "squirrel",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
            "multi-gpu": [
                "hub3",
                "hub3-remote",
                "ffcv",
                "pytorch",
                "squirrel",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
            "filtering": [
                "hub",
                "hub-remote",
                "hub3",
                "hub3-remote",
                "pytorch",
                "torchdata",
                "webdataset",
                "webdataset-remote",
            ],
        },
        "filtering_classes": ["dog", "truck"],
        "is_cutoff_run_model": [True],
        "cutoff": 10,
    },
    "coco": {
        "batch_size": [1, 2],  # [2, 8, 32, 64, 128], #, 32, 64, 128],
        "workers": [0, 1],  # , 8, 16, 32],
        "libraries": {
            "single-gpu": [
                # "hub", # works
                # "hub-remote",
                # "hub3",
                # "pytorch", # works
                "squirrel", # works
                # "torchdata", # works
                # "webdataset", # works
                # "webdataset-remote", # works
            ],
            "multi-gpu": [
                # "hub3"
                # "pytorch", # works
                "squirrel", # works
                # "torchdata", # works
                # "webdataset", # works
                # "webdataset-remote", # works
            ],
            "filtering": [
                # "pytorch", # don't run, too slow
                # "torchdata", # works 
                # "webdataset",
                # "webdataset-remote",
                # "hub", # works
                "hub-remote",
                # "hub3" # not working yet
            ],
        },
        "is_cutoff_run_model": [True],
        "filtering_classes": ["pizza", "couch", "cat"],
        "cutoff": 5,
    },
}
