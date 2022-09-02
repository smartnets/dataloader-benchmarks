# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from distutils.util import strtobool
from matplotlib.lines import Line2D
from experiments.configuration import EXPERIMENTS
from src.utils.general import config_to_bool
from src.config import settings as st

plt.style.use(["science", "no-latex", "ieee", "std-colors"])
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


PATH = (
    Path(st.ROOT_PATH_FOR_DYNACONF) / "downloaded_results"
)  # folder needs to be created
p2 = Path(st.ROOT_PATH_FOR_DYNACONF) / "plots"
p2.mkdir(parents=True, exist_ok=True)
# %%

rename_x = {"num_workers": "Number of Workers", "batch_size": "Batch Size"}

target = "avg_speed"
target_label = "Average Speed [#/s]"


def compute_average_speed(runtimes: list, batch_size: int):
    """
    runtimes: list of the batch times for each worker.
    """

    speed_list = [np.mean([batch_size / t for t in rt[1:]]) for rt in runtimes]
    return np.sum(speed_list)


def add_single_runtimes(d, exp):

    accounted_time = 0
    all_runtimes = exp["0"]["all_runtimes"]
    train_loader = exp["0"]["dataloader_start_time"]["train"]
    total_time = exp["0"]["total_training_time"]

    for t, v in enumerate(all_runtimes):
        d[f"time_batch_{t}"] = v
        accounted_time += v

    d["time_train_loader"] = train_loader[1] - train_loader[0]
    accounted_time += train_loader[1] - train_loader[0]
    d["remaining_time"] = total_time - accounted_time


def extract_metrics(exp):

    proc = dict()
    proc["dataset"] = exp["DATASET"]
    proc["num_workers"] = int(exp["NUM_WORKERS"])
    proc["batch_size"] = int(exp["BATCH_SIZE"])
    proc["rep"] = exp["REP"]
    proc["library"] = exp["LIBRARY"]
    proc["remote"] = config_to_bool(exp["REMOTE"])
    proc["cutoff"] = exp["CUTOFF"]

    if config_to_bool(exp["FILTERING"]):
        mode = "filtering"
    elif config_to_bool(exp["DISTRIBUTED"]):
        mode = "distributed"
    else:
        mode = "default"
    proc["mode"] = mode

    runtimes = [exp["0"]["all_runtimes"]]
    if "1" in exp:
        runtimes += [exp["1"]["all_runtimes"]]
    proc["avg_speed"] = compute_average_speed(runtimes, proc["batch_size"])

    add_single_runtimes(proc, exp)

    if not proc["num_workers"] in EXPERIMENTS[proc["dataset"]]["workers"]:
        return None
    if not proc["batch_size"] in EXPERIMENTS[proc["dataset"]]["batch_size"]:
        return None

    return proc


# %%
experiments = []
for file in PATH.rglob("*.json"):
    with open(file, "r") as fh:
        data = json.load(fh)
    proc = extract_metrics(data)
    if proc:
        experiments += [proc]
df = pd.DataFrame(experiments)

df = df[(df["cutoff"] == 10)].dropna(axis=1, how="all")


pal = dict((l, c) for l, c in zip(df["library"].unique(), COLORS))

# %%


datasets = ["cifar10", "random", "coco"]
modes = ["default", "distributed", "filtering"]

for ds in datasets:
    for m in modes:
        df_ = df[(df["dataset"] == ds) & (df["mode"] == m) & (df["remote"] == False)]

        for x in ["num_workers", "batch_size"]:
            try:
                df_x = (
                    df_.groupby([x, "library", "rep"])["avg_speed"].max().reset_index()
                )
                title = f"{ds}_{m}_{x}"

                fig, ax = plt.subplots()
                sns.barplot(
                    data=df_x,
                    x=x,
                    y="avg_speed",
                    hue="library",
                    dodge=True,
                    ax=ax,
                    palette=pal,
                )
                # ax.set_title(title)
                ax.set_xlabel(rename_x[x])
                ax.set_ylabel(target_label)
                ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc="lower center")
                fig.savefig(p2 / (title + ".jpg"))

            except Exception as e:
                print(e)
                print("Failed", ds, m)


# %%


ds = "random"
nw = 0
rep = 1
bs = 64
m = "default"
df_2 = df.copy()
df_2 = df_2.query("dataset == @ds and num_workers == @nw and batch_size == @bs")
df_2 = df_2.query("rep == @rep and remote == 0 and mode == @m")


ylabels = []
times = []
colors = []
for i, r in df_2.iterrows():
    ylabels.append(r.library)

    colors = [COLORS[0]]

    t_ = [r.time_train_loader]
    for j in range(r.cutoff):
        t_ += [r[f"time_batch_{j}"]]
        colors += [COLORS[1]]
    t_ += [r.remaining_time]
    colors += [COLORS[2]]
    times.append(t_)

# %%
fig, ax = plt.subplots()
from matplotlib.patches import Rectangle

h = 0.1
for i, exp in enumerate(times):
    x = 0
    y = i * h * 2
    for j, t in enumerate(exp):
        rec = Rectangle((x, y), t, h, linewidth=1, fc=colors[j], ec="k")

        if j > 0 and j < len(exp) - 1:

            rx, ry = rec.get_xy()
            cx = rx + rec.get_width() / 2.0
            cy = ry + rec.get_height() / 2.0

            print(cx)

            if rec.get_width() > 0.03:
                ax.annotate(
                    str(j - 1),
                    (cx, cy),
                    color="k",
                    weight="bold",
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        x += t
        ax.add_patch(rec)


max_x = max([sum(x) for x in times])
ax.set_xlim([0, max_x * 1.05])
ax.set_ylim([0, 2 * h * (len(times))])


custom_lines = [
    Line2D([0], [0], color=COLORS[0], lw=4),
    Line2D([0], [0], color=COLORS[1], lw=4),
    Line2D([0], [0], color=COLORS[2], lw=4),
]
ax.legend(
    custom_lines,
    ["Dataloader Init", "Batch", "Wrap Up"],
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
    ncol=3,
)

ax.set_xlabel("Time [s]")

ax.set_yticks([(h / 2) + (h * i * 2) for i in range(len(times))])
ax.set_yticklabels(ylabels)
fig.show()
fig.savefig(p2 / "closer_look.jpg")

# %%

ds = "random"
x = "library"
hue = "mode"
df_2 = df.copy()
df_2 = df_2.query("dataset == @ds and mode != 'filtering'")
df_2 = df_2.query("remote == 0")
df_2 = df_2[["num_workers", "batch_size", "rep", "library", "mode", "avg_speed"]]

g = (
    df_2.groupby([x, hue])[target]
    .max()
    .groupby(level=0)
    .filter(lambda y: len(y) > 1)
    .reset_index()
    .sort_values(x)
)

# %%

N = int(g[x].nunique())
x_range = np.arange(N)
width = 0.35

s_vals = g[g[hue] == "default"][target].values
m_vals = g[g[hue] == "distributed"][target].values

fig, ax = plt.subplots()
r1 = ax.bar(x_range - width / 2, s_vals, width=width, label="Single GPU")
r2 = ax.bar(x_range + width / 2, m_vals, width=width, label="Multi GPU")
ax.legend()
ax.set_xticks(x_range, list(g[x].unique()))
ax.set_ylabel(target_label)
ax.tick_params(axis="x", labelrotation=10)


change = (m_vals - s_vals) / s_vals
change_str = [f"{c:.1%}" for c in change]
ax.bar_label(r2, padding=0, labels=change_str, label_type="edge")
fig.savefig(p2 / f"ddp_gains_{ds}.jpg")
# fig_name = f"ddp_gains_{dataset}.png"

# ax.set_title(title)
# fig.savefig(fig_name)

# %%

df_interest = pd.read_csv("interest_over_time.csv", header=1)
df_interest = df_interest.set_index("Week")
df_interest.index = pd.to_datetime(df_interest.index)
df_interest.columns = [c.replace(": (Worldwide)", "") for c in df_interest.columns]
# %%
fig, ax = plt.subplots()
df_interest.plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("Interest over Time [0-100]")
fig.savefig(p2 / "interest.jpg")
# %%

ds = "random"
x = "library"
hue = "mode"
df_2 = df.copy()
df_2 = df_2.query("dataset == @ds and mode != 'distributed'")
df_2 = df_2.query("remote == 0")
df_2 = df_2[["num_workers", "batch_size", "rep", "library", "mode", "avg_speed"]]

g = (
    df_2.groupby([x, hue])[target]
    .max()
    .groupby(level=0)
    .filter(lambda y: len(y) > 1)
    .reset_index()
    .sort_values(x)
)


N = int(g[x].nunique())
x_range = np.arange(N)
width = 0.35

s_vals = g[g[hue] == "default"][target].values
m_vals = g[g[hue] == "filtering"][target].values

fig, ax = plt.subplots()
r1 = ax.bar(x_range - width / 2, s_vals, width=width, label="Default")
r2 = ax.bar(x_range + width / 2, m_vals, width=width, label="Filtering")
ax.legend()
ax.set_xticks(x_range, list(g[x].unique()))
ax.set_ylabel(target_label)
ax.tick_params(axis="x", labelrotation=10)


change = (m_vals - s_vals) / s_vals
change_str = [f"{c:.1%}" for c in change]
ax.bar_label(r2, padding=0, labels=change_str, label_type="edge")
fig.savefig(p2 / f"filtering_gains_{ds}.jpg")

# %%
