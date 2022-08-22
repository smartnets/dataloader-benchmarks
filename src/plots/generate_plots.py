# %%
import seaborn as sns
import matplotlib.pyplot as plt
from src.plots.collect_experiments import get_experiment_df, GROUPS
from functools import partial
import matplotlib as mp
from matplotlib import cm
import numpy as np

# %%
# CORE_GROUP = ["LIBRARY", "DATASET", "DISTRIBUTED", "REMOTE"]
target = "proc_samples_per_sec"
target_label = "Processed Samples [#/s]"
hue = "LIBRARY"

# %%


def filter_dataset(df, dataset, distributed, run_model):
    return df[
        (df["DATASET"] == dataset)
        & (df["DISTRIBUTED"] == distributed)
        & (df["IS_CUTOFF_RUN_MODEL"] == run_model)
    ].copy()


# %%
df = get_experiment_df()

# %%


def correlation_plot(df, dataset):

    df_ = df[df["CUTOFF"] < 0]
    df_ = df_[df_["DATASET"] == dataset]

    fig, ax = plt.subplots()
    df_.plot.scatter(x="total_run_time", y="proc_samples_per_sec", ax=ax)
    ax.set_xlabel("Total Running Time (desired)")
    ax.set_ylabel("Processed samples per second (observed)")
    ax.set_title("Correlation between metrics of interest")

    fig.savefig(f"correlation_{dataset}.png")


# %%
def generate_plot(df, params):

    p = params

    df = filter_dataset(df, p["dataset"], p["is_distributed"], p["run_model"])
    df = df.groupby(GROUPS + [p["x_axis"]])[p["target"]].max().reset_index()

    cmap = cm.get_cmap("tab10")

    fig, ax = plt.subplots()
    for ii, hue_ in enumerate(df[p["hue"]].unique()):

        style = "--" if "s3" in hue_ else "-"
        color = cmap.colors[ii]
        t = df[df[p["hue"]] == hue_]
        ax.plot(
            t[p["x_axis"]],
            t[p["target"]],
            label=t["NAME"].iloc[0],
            marker="*",
            c=color,
            linestyle=style,
        )

    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="lower center", borderaxespad=0, ncol=3)
    ax.set_xlabel(p["x_axis_label"])
    ax.set_ylabel(p["target_label"])

    title = p["title"] + ":"
    title += " 2+GPU," if p["is_distributed"] else "1GPU,"
    title += " RM" if p["run_model"] else "Not RM"

    ax.set_title(title)

    fig_name = "dist" if p["is_distributed"] else "not-dist"
    fig_name += "_"
    fig_name += "run" if p["run_model"] else "not-run"
    fig_name += "_"
    fig_name += p["x_axis"].lower()
    fig_name += f"_{p['dataset']}"
    fig_name += ".png"

    fig.tight_layout()
    fig.savefig(fig_name)

    return fig, ax
    # fig, ax = plt.subplot()


def plot_ddp_gains(df, params):
    p = params
    x = p["x"]
    dataset = p["dataset"]
    title = p["title"]
    target = p["target"]
    target_label = p["target_label"]
    hue = p["hue"]
    cols = [x, hue]

    df_ = df[df["DATASET"] == dataset].copy()
    df_ = df_[df_["FILTERING"] == 0]
    df_ = df_[df_["IS_CUTOFF_RUN_MODEL"] == True]

    g = (
        df_.groupby(cols)[target]
        .max()
        .groupby(level=0)
        .filter(lambda x: len(x) > 1)
        .reset_index()
        .sort_values(x)
    )
    N = int(g[x].nunique())
    x_range = np.arange(N)
    width = 0.35

    s_vals = g[g[hue] == 0][target].values
    m_vals = g[g[hue] == 1][target].values

    fig, ax = plt.subplots()
    r1 = ax.bar(x_range - width / 2, s_vals, width=width, label="Single GPU")
    r2 = ax.bar(x_range + width / 2, m_vals, width=width, label="Multi GPU")
    ax.legend()
    ax.set_xticks(x_range, list(g[x].unique()))
    ax.set_ylabel(target_label)

    change = (m_vals - s_vals) / s_vals
    change_str = [f"{c:.1%}" for c in change]
    ax.bar_label(r2, padding=0, labels=change_str, label_type="edge")

    fig_name = f"ddp_gains_{dataset}.png"

    ax.set_title(title)
    fig.savefig(fig_name)

    return fig, ax


def plot_filtering_gains(df, params):
    p = params
    x = p["x"]
    dataset = p["dataset"]
    title = p["title"]
    target = p["target"]
    target_label = p["target_label"]
    hue = p["hue"]
    cols = [x, hue]

    df_ = df[df["DATASET"] == dataset].copy()
    df_ = df_[df_["DISTRIBUTED"] == False]
    df_ = df_[df_["IS_CUTOFF_RUN_MODEL"] == True]

    g = (
        df_.groupby(cols)[target]
        .max()
        .groupby(level=0)
        .filter(lambda x: len(x) > 1)
        .reset_index()
        .sort_values(x)
    )
    N = int(g[x].nunique())
    x_range = np.arange(N)
    width = 0.35

    s_vals = g[g[hue] == False][target].values
    m_vals = g[g[hue] == True][target].values

    fig, ax = plt.subplots()
    r1 = ax.bar(x_range - width / 2, s_vals, width=width, label="No Filtering")
    r2 = ax.bar(x_range + width / 2, m_vals, width=width, label="With Filtering")
    ax.legend(bbox_to_anchor=(0.2, 1.15))
    ax.set_xticks(x_range, list(g[x].unique()))
    ax.set_ylabel(target_label)

    change = (m_vals - s_vals) / s_vals
    change_str = [f"{c:.1%}" for c in change]
    ax.bar_label(r2, padding=0, labels=change_str, label_type="edge")

    fig_name = f"filtering_gains_{dataset}.png"

    ax.set_title(title)
    fig.savefig(fig_name)

    return fig, ax


for ds in ["cifar10", "random"]:

    correlation_plot(df, ds)

    # %%
    # Case 2
    params = {
        "dataset": ds,
        "is_distributed": False,
        "run_model": True,
        "target": target,
        "target_label": target_label,
        "x_axis": "NUM_WORKERS",
        "x_axis_label": "Number of Workers",
        "hue": "NAME",
        "title": "Impact of workers in speed",
    }
    res = generate_plot(df, params)

    # %%
    # Case 3
    params = {
        "dataset": ds,
        "is_distributed": False,
        "run_model": True,
        "target": target,
        "target_label": target_label,
        "x_axis": "BATCH_SIZE",
        "x_axis_label": "Batch Size",
        "hue": "NAME",
        "title": "Impact of batch_size in speed",
    }
    res = generate_plot(df, params)

    # %%

    # %%
    params = {
        "dataset": ds,
        "x": "NAME",
        "target": target,
        "target_label": target_label,
        "hue": "DISTRIBUTED",
        "title": "Speed gains with multi GPU",
    }
    res = plot_ddp_gains(df, params)

    params = {
        "dataset": ds,
        "x": "NAME",
        "target": target,
        "target_label": target_label,
        "hue": "FILTERING",
        "title": "Speed gains from Filtering",
    }
    res = plot_filtering_gains(df, params)
