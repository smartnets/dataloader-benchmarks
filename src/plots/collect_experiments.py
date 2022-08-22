# %%
import pandas as pd
import json
from src.config import settings as st
from pathlib import Path
from distutils.util import strtobool

# %%
GROUPS = [
    "DATASET",
    "LIBRARY",
    "DISTRIBUTED",
    "REMOTE",
    "IS_CUTOFF_RUN_MODEL",
    "NAME",
    "CUTOFF",
    "FILTERING",
    "total_run_time",
    "proc_samples",
]

EXTRA = ["NUM_WORKERS", "BATCH_SIZE"]


def get_experiment_df():
    params_path = Path(st.downloaded_results_dir)

    json_list = []
    for path in params_path.rglob("*parameters.json"):
        with open(path, "r") as fh:
            j = json.load(fh)
        j["timestamp"] = path.parent.stem
        json_list.append(j)

    df = pd.DataFrame.from_records(json_list)
    # df["FILTERING"] = df["FILTERING"].map(
    #     lambda x: len(x) if isinstance(x, list) else 0
    # )

    # df["FILTERING"] = df["FILTERING"].astype(str)

    for col in ["REMOTE", "DISTRIBUTED", "IS_CUTOFF_RUN_MODEL", "FILTERING"]:
        df[col] = df[col].astype(str).map(strtobool)

    df["NAME"] = df.apply(
        lambda x: x["LIBRARY"] + "-s3" if x["REMOTE"] else x["LIBRARY"], axis=1
    )
    # return df

    res = (
        df.sort_values("timestamp")
        .groupby(GROUPS + EXTRA)
        .tail(1)[GROUPS + EXTRA + ["proc_samples_per_sec"]]
        .copy()
    )

    return res


# %%
df = get_experiment_df()
# %%
