# %%
import json
from tkinter import W
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from distutils.util import strtobool
from matplotlib.lines import Line2D
from src.config import settings as st
from src.utils.general import config_to_bool

plt.style.use(["science", "no-latex", "ieee", "std-colors"])
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


PATH = (
    Path(st.ROOT_PATH_FOR_DYNACONF) / "remote_experiments_results"  # needs to be set up
)  # folder needs to be created
p2 = Path(st.ROOT_PATH_FOR_DYNACONF) / "plots"
p2.mkdir(parents=True, exist_ok=True)


# %%

rename_x = {"num_workers": "Number of Workers", "batch_size": "Batch Size"}

target = "total_time"
target_label = "Total Running Time"


def extract_metrics(exp):

    proc = dict()
    proc["dataset"] = exp["DATASET"]
    proc["library"] = exp["LIBRARY"]
    proc["total_time"] = float(exp["0"]["total_training_time"])

    if config_to_bool(exp["FILTERING"]):
        mode = "filtering"
    elif config_to_bool(exp["DISTRIBUTED"]):
        mode = "distributed"
    else:
        mode = "default"
    proc["mode"] = mode

    remote = config_to_bool(exp["REMOTE"])
    aws = config_to_bool(exp["IS_AWS"])

    if remote and aws:
        source = "AWS"
    elif remote:
        source = "MinIO"
    else:
        source = "Local"

    proc["source"] = source

    return proc


# %%
experiments = []
for file in PATH.rglob("*.json"):
    with open(file, "r") as fh:
        data = json.load(fh)
    proc = extract_metrics(data)
    experiments += [proc]
df = pd.DataFrame(experiments)
df["library"] = df["library"].replace({"deep_lake": "Deep Lake"})

# %%
fig, ax = plt.subplots()

sns.barplot(data=df, x="source", hue="library", y="total_time")
ax.set_xlabel("Data Source")
ax.set_ylabel("Total Running Time [s]")

# %%
x = "source"
hue = "library"
target = "total_time"

order = {"Local": 0, "AWS": 1, "MinIO": 2}

df = df.sort_values(by=["source"], key=lambda x: x.map(order))

N = int(df[x].nunique())
x_range = np.arange(N)
width = 0.2

libs = sorted(df[hue].unique())
# lib_names = ["Hub", "Deep Lake", "Webdataset"]

fig, ax = plt.subplots()
for i, lib in enumerate(libs):

    vals = df[df[hue] == lib][target].values
    r = ax.bar(x_range + width * (i - 1), vals, width=width, label=libs[i])

    c = (vals - vals[0]) / vals[0]
    c_str = [f"{c:.0%}" if c > 0 else "" for c in c]
    ax.bar_label(r, padding=0, labels=c_str, label_type="center", fontsize=5, color="w")


ax.legend()
ax.set_xticks(x_range, list(df[x].unique()))
ax.set_ylabel(target_label)
ax.set_xlabel("Source")
fig.savefig(p2 / "source_gains.jpg")

# %%
