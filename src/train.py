# %%
from src.profiling.metric_logger import MetricLogger
from src.config import settings as st
from src.utils.general import config_to_list, should_run_test, config_to_bool


def setup(model_, loader_, rank: int = 0, run_id: str = "0"):

    distributed = config_to_bool(st.distributed)
    filtering = config_to_bool(st.filtering)
    filtering_classes = config_to_list(st.filtering_classes)
    device = f"cuda:{rank}" if distributed else st.device

    loader = loader_(
        remote=st.remote,
        filtering=filtering,
        filtering_classes=filtering_classes,
        distributed=distributed,
        world_size=st.world_size,
        rank=rank,
    )

    params = {
        "profiling_epochs": st.profiling_enabled,
        "profiling_steps": st.profiling_epochs_per_round,
        "automatic_gpu_transfer": loader.automatic_gpu_transfer,
        "cutoff": int(st.cutoff),
        "distributed": distributed,
    }

    loader_kwargs = {
        "batch_size": int(st.batch_size),
        "num_workers": int(st.num_workers),
    }
    metric_logger = MetricLogger(run_id, rank)
    metric_logger.start_side_collectors()

    loaders = {}
    metric_logger.log_time_loaders("train")
    loaders["train"] = loader.get_train_loader(**loader_kwargs)
    metric_logger.log_time_loaders("train")

    if int(st.cutoff) < 0:
        for ds, _get in zip(
            ["val", "test"], [loader.get_val_loader, loader.get_test_loader]
        ):
            if ds == "test" and not should_run_test():
                continue
            metric_logger.log_time_loaders(ds)
            loaders[ds] = _get(**loader_kwargs)
            metric_logger.log_time_loaders(ds)

    model = model_()

    model.initialize(rank, distributed=distributed)

    return model, loaders, params, metric_logger
