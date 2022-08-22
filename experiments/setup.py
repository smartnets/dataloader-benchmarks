import os


def configure_env(
    dataset,
    library,
    num_workers,
    cutoff,
    batch_size,
    is_cutoff_run_model,
    multi_gpu,
    filtering,
    filtering_classes,
):

    print(batch_size, num_workers, library)

    os.environ["DYNACONF_LIBRARY"] = library.replace("-remote", "")
    os.environ["DYNACONF_DATASET"] = dataset
    os.environ["DYNACONF_BATCH_SIZE"] = str(batch_size)
    os.environ["DYNACONF_NUM_WORKERS"] = str(num_workers)
    os.environ["DYNACONF_REMOTE"] = str("-remote" in library)
    os.environ["DYNACONF_CUTOFF"] = str(cutoff)
    os.environ["DYNACONF_DISTRIBUTED"] = str(multi_gpu)
    os.environ["DYNACONF_IS_CUTOFF_RUN_MODEL"] = str(is_cutoff_run_model)
    os.environ["DYNACONF_FILTERING"] = str(filtering)
    os.environ["DYNACONF_FILTERING_CLASSES"] = str(filtering_classes)

    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
