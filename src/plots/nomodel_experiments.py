# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from distutils.util import strtobool
from matplotlib.lines import Line2D
from src.config import settings as st
from src.utils.general import config_to_bool
from scipy.stats import pearsonr

plt.style.use(["science", "no-latex", "ieee", "std-colors"])
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


PATH = (
    Path(st.ROOT_PATH_FOR_DYNACONF) / "results_nomodel"  # needs to be set up
)  # folder needs to be created
p2 = Path(st.ROOT_PATH_FOR_DYNACONF) / "plots"
p2.mkdir(parents=True, exist_ok=True)


# %%

rename_x = {"num_workers": "Number of Workers", "batch_size": "Batch Size"}

target = "total_time"
target_label = "Total Running Time"


def compute_average_speed(runtimes: list, batch_size: int):
    """
    runtimes: list of the batch times for each worker.
    """

    speed_list = [np.mean([batch_size / t for t in rt[1:]]) for rt in runtimes]
    return np.sum(speed_list)


def extract_metrics(exp):

    proc = dict()
    proc["dataset"] = exp["DATASET"]
    proc["library"] = exp["LIBRARY"]
    proc["num_workers"] = int(exp["NUM_WORKERS"])
    proc["batch_size"] = int(exp["BATCH_SIZE"])
    proc["epochs"] = int(exp["NUM_EPOCHS"])
    proc["total_time"] = float(exp["0"]["total_training_time"])

    proc["run_model"] = bool(config_to_bool(exp["IS_CUTOFF_RUN_MODEL"]))

    runtimes = [exp["0"]["all_runtimes"]]
    if "1" in exp:
        runtimes += [exp["1"]["all_runtimes"]]
    proc["avg_speed"] = compute_average_speed(runtimes, proc["batch_size"])

    return proc


# %%
experiments = []
for file in PATH.rglob("*.json"):
    with open(file, "r") as fh:
        data = json.load(fh)
    proc = extract_metrics(data)
    experiments += [proc]
df = pd.DataFrame(experiments)
df["library"] = df["library"].replace(
    {
        "hub3": "Deep Lake",
        "webdataset": "Webdataset",
        "ffcv": "FFCV",
        "torchdata": "Torchdata",
        "pytorch": "Pytorch",
        "squirrel": "Squirrel",
        "hub": "Hub",
    }
)


# %%
df["run_model"] = df["run_model"].map({True: "Loading & Fwd & Bkw", False: "Loading"})

# %%
fig, ax = plt.subplots()

sns.barplot(data=df, x="library", hue="run_model", y="avg_speed")

ax.set_xlabel("Library")
ax.set_ylabel("Total Running Time [s]")
ax.tick_params(axis="x", labelrotation=20)
ax.set_yscale("log")
ax.legend()

fig.savefig(p2 / "run_model_comparission_random.jpg")
# %%


# %%
df_ = df[df["run_model"] == "Loading & Fwd & Bkw"]
fig, ax = plt.subplots()
sns.regplot(data=df_, x="total_time", y="avg_speed", ax=ax)
ax.set_xlabel("Total Training Time [s]")
ax.set_ylabel("Average Speed [#/s]")

fig.savefig(p2 / "correlation_speed_time.jpg")

# %%

# %%
x = df_["avg_speed"].values
y = df_["total_time"].values

res = pearsonr(x, y)
print(res)
# %%
