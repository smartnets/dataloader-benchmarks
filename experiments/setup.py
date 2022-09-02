import os
import datetime


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
    rep,
    filename,
    epochs,
):

    now = datetime.datetime.now().isoformat()
    iteration_text = (
        f"{rep:<2}, {batch_size:<3}, {num_workers:<2}, {library:<15}: {now}"
    )
    print(iteration_text)
    with open(filename, "a") as fh:
        fh.write(iteration_text + "\n")

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
    os.environ["DYNACONF_REP"] = str(rep)
    os.environ["DYNACONF_NUM_EPOCHS"] = str(epochs)

    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
