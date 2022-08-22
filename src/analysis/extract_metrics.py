# %%
import subprocess
import json
import pandas as pd
from pathlib import Path
import numpy as np

# %%
def get_times_stats(path):
    with open(path / "times.json", "r") as fh:
        all_times = json.load(fh)
    step_times = []
    start_times = []

    times = all_times["epochs"]
    epochs = [k for k in times.keys() if k.isnumeric()]
    for e in epochs:
        steps = [k for k in times[e].keys() if k.isnumeric()]

        for i in range(len(steps)):
            step = times[e][str(i)]
            elapsed = step["end"] - step["start"]

            if i == 0:
                prev_end = times[e]["begin"]
            else:
                prev_end = times[e][str(i - 1)]["end"]
            start_time = step["start"] - prev_end

            step_times.append(elapsed)
            start_times.append(start_time)

    start_times = np.array(start_times)
    step_times = np.array(step_times)

    total_step = step_times.sum()
    total_start = start_times.sum()
    total_run = times["end"]

    L = all_times["loaders"]

    TRL = L.get("train", [0, 0])
    VAL = L.get("val", [0, 0])
    TEL = L.get("test", [0, 0])

    results = {
        "total_step_time": total_step,
        "total_start_time": total_start,
        "total_run_time": total_run,
        "loader_train_start_time": TRL[1] - TRL[0],
        "loader_val_start_time": VAL[1] - VAL[0],
        "loader_test_start_time": TEL[1] - TEL[0],
    }

    return results


# %%


def get_cpu_usage(path):
    with open(path / "cpu_usage.txt", "r") as fh:
        X = fh.readlines()
    df_cpu = pd.DataFrame([y.strip().split(",") for x in X for y in x.split("|")])
    df_cpu.columns = ["PID", "CPU", "Memory"]
    df_cpu[["CPU", "Memory"]] = df_cpu[["CPU", "Memory"]].astype(float)
    parent_pid = X[0].split(",")[0]
    df_cpu_main = df_cpu[df_cpu["PID"] == parent_pid]

    qs = [0.01, 0.25, 0.5, 0.75, 0.95]
    cpu_usage = np.quantile(df_cpu["CPU"].values, qs)
    memory_usage = np.quantile(df_cpu["Memory"].values, qs)

    result = {}
    for q, c, m in zip(qs, cpu_usage, memory_usage):
        result[f"cpu-usage-{q:.2f}"] = c
        result[f"memory-usage-{q:.2f}"] = m

    return result


# %%
def get_gpu_usage(path):
    df_gpu = pd.read_csv(path / "gpu_usage.txt", header=None)
    df_gpu.columns = ["time", "id", "ut-gpu", "ut-memory"]
    df_gpu["time"] = pd.to_datetime(df_gpu["time"])
    df_gpu["ut-gpu"] = df_gpu["ut-gpu"].apply(lambda x: int(x[:-2]))
    df_gpu["ut-memory"] = df_gpu["ut-memory"].apply(lambda x: int(x[:-2]))

    qs = [0.01, 0.25, 0.5, 0.75, 0.95]
    gpu_metrics = np.quantile(df_gpu["ut-gpu"].values, qs)
    result = {}
    for q, m in zip(qs, gpu_metrics):
        result[f"gpu-usage-{q:.2f}"] = m
    return result


# %%
def get_gpu_name():

    res = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], shell=False
    )
    result = {"gpu-name": res.decode("utf-8").strip().split("\n")[1]}
    return result


# %%

# p = Path("/home/worker/workspace/results/cifar10/pytorch")

# %%


def collect_metrics(path):

    time = get_times_stats(path)
    gpu = get_gpu_usage(path)
    cpu = get_cpu_usage(path)
    gpu_name = get_gpu_name()
    final = {**time, **gpu, **cpu, **gpu_name}
    return final


# %%
# A = collect_metrics(p)
# %%
