import torch
from pathlib import Path


def get_profiler(output_file: Path, **kwargs):

    if not isinstance(output_file, Path):
        output_file = Path(output_file)

    profiling_steps = kwargs.get("profiling_steps", 5)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_file / "traces"),
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=profiling_steps, repeat=1
        ),
    )
    return profiler
