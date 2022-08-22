# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import settings as st


def plot_loss(dataset, library):

    path = st.local_results_dir / dataset / library

    df = pd.read_csv(path / "loss.txt", header=None)
    EPOCHS = df.shape[0]

    if EPOCHS < 10:
        SPACE_TICKS = 1
    elif EPOCHS < 50:
        SPACE_TICKS = 5
    elif EPOCHS < 200:
        SPACE_TICKS = 200
    else:
        SPACE_TICKS = 100
    # %%
    fig, ax = plt.subplots()
    ax.plot(range(EPOCHS), df[0].values, label="Train")
    ax.plot(range(EPOCHS), df[1].values, label="Validation")
    ax.set_xlabel("#Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(range(0, EPOCHS, SPACE_TICKS))
    ax.set_xticklabels(range(0, EPOCHS, SPACE_TICKS))
    ax.legend()
    fig.savefig(path / "loss.jpg", bbox_inches="tight", dpi=150)


def plot_times(dataset, library, filtering=False):

    path = st.local_results_dir / dataset / library
    if filtering:
        path /= "filtering"
    # %%
    with open(path / "times.json", "r") as fh:
        times = json.load(fh)

    fig, axs = plt.subplots(2, 1, figsize=(16, 12))

    for e in [0, 1]:

        step_times = []
        start_times = []
        e = str(e)
        steps = [k for k in times[e].keys() if k != "begin" and k != "end"]

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

        ax = axs[int(e)]
        ax.set_yscale("log")
        ax.plot(step_times, label="Time per step")
        ax.plot(start_times, label="Time in Loader")
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("# Step")
        ax.set_title(f"Running times for epoch {e}")
        ax.legend()

    fig.suptitle(f"Total running time: {times['end']:.2f} s")
    fig.tight_layout()
    fig.savefig(path / "times.jpg", bbox_inches="tight", dpi=150)
    # %%
