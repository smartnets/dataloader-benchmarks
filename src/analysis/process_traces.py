# %%
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import settings as st

# %%


def get_first_gpu_usage(dataframe, step):
    """
    Time between start of the step and first
    GPU activity, in seconds
    """

    first_gpu_usage = dataframe[dataframe["device"] == 0]["ts"].min()
    time_to_gpu = (first_gpu_usage - step["start"]).total_seconds()
    return time_to_gpu


def get_total_time_gpu(dataframe, step):
    """
    Get total running time in GPU (in seconds)
    """
    return dataframe[dataframe["device"] == 0]["dur"].sum() / 1e6


def get_total_time_loader(dataframe, step):
    """
    Get total time spent in dataloader  in seconds
    """
    total_time = (
        dataframe[
            (dataframe["Call stack"].str.contains("__next__").fillna(False))
            & (dataframe["Call stack"].str.contains("dataloader.py").fillna(False))
        ]["dur"]
        .fillna(0)
        .max()
    )
    return total_time / 1e6


# dataset = "cifar10"
# library = "pytorch"


def plot_traces(dataset, library):
    # %%
    # dataset = "cifar10"
    # library = "hub-remote"

    # # %%
    # if True:
    # %%
    path = Path(st.local_results_dir) / dataset / library
    # %%

    print("Entre")
    files = path.glob("traces/*")

    for file in files:

        with open(file, "r") as fh:
            data = json.load(fh)
        # %%
        traces = data["traceEvents"]
        for t in traces:
            if "args" in t:
                t.update(t["args"])
                del t["args"]
        # %%
        df = pd.DataFrame(traces)
        df["ts"] = pd.to_datetime(df["ts"], unit="us")

        steps = []
        for row in df[df["name"].str.contains("ProfilerStep#")].itertuples():
            steps.append(
                {
                    "index": row.Index,
                    "name": row.name,
                    "start": row.ts,
                    "end": row.ts + pd.Timedelta(microseconds=row.dur),
                    "elapsed_time": row.dur / 1e6,
                }
            )

        # %%

        for step in steps:

            start, end = step["start"], step["end"]
            df_step = df[(df["ts"] >= start) & (df["ts"] <= end)]

            first_gpu_usage = get_first_gpu_usage(df_step, step)
            total_time_gpu = get_total_time_gpu(df_step, step)
            total_time_loader = get_total_time_loader(df_step, step)

            step["time_to_gpu"] = first_gpu_usage
            step["time_in_gpu"] = total_time_gpu
            step["time_in_loader"] = total_time_loader

        # %%
        # %%

        df_plot = pd.DataFrame(steps).sort_values("start")
        df_plot.head()

        # %%
        fig, ax = plt.subplots()
        width = 0.5
        elapsed = df_plot["elapsed_time"]
        gpu_time = df_plot["time_in_gpu"] / elapsed * 100
        loader_time = df_plot["time_in_loader"] / elapsed * 100

        total = np.ones(len(elapsed)) * 100

        labels = [str(i) for i in range(df_plot.shape[0])]

        ax.bar(labels, total, width, label="Total Time", color="#D32E5EFF", alpha=0.6)

        ax.bar(
            labels,
            loader_time,
            width,
            label="Time in Loader",
            color="#A59C94FF",
            alpha=0.8,
        )
        ax.bar(
            labels,
            gpu_time,
            width,
            bottom=loader_time,
            label="Time in GPU",
            color="#AE0E36FF",
        )

        # ax.set_yscale("log")
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
        ax.set_ylabel("Percentage of time in Step [%]")
        ax.set_xlabel("# Step")
        fig.tight_layout()
        fig.savefig(file.with_suffix(".jpg"))

        # %%
