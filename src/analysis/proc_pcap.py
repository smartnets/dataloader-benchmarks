# %%
import datetime
import subprocess
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scapy.all import *

from src.config import settings as st

# %%


def pcap_to_csv(path):

    TSHARK_SMI_ARGS = [
        "/usr/bin/tshark",
        "-T",
        "fields",
        "-n",
        "-r",
        str(path / "dump.pcap"),
        "-E",
        "separator=|",
    ]
    for field in [
        "ip.src",
        "ip.dst",
        "frame.time",
        "frame.len",
        "_ws.col.Protocol",
        "data.data",
    ]:
        TSHARK_SMI_ARGS.append("-e")
        TSHARK_SMI_ARGS.append(field)

    with open(path / "pcap.csv", "w") as outfile:
        pid = subprocess.Popen(TSHARK_SMI_ARGS, stdout=outfile)
    pid.wait()


def plot_network_usage(dataset, library):
    # %%
    # dataset = "cifar10"
    # library = "hub-remote"

    # # %%
    # if True:
    # %%
    path = Path(st.local_results_dir) / dataset / library
    # %%
    pcap_to_csv(path)

    # %%
    df_raw = pd.read_csv(path / "pcap.csv", sep="|", header=None)
    df_raw.columns = ["src", "dest", "time", "load", "protocol", "payload"]
    df_raw["load"] = df_raw["load"] / 1024 / 1024  # byte to mb

    times = df_raw["time"]
    times = [datetime.strptime(t[:-7], "%b %d, %Y %H:%M:%S.%f") for t in times]
    df_raw["time"] = pd.to_datetime(times)
    # %%

    data_stats = df_raw.groupby(["src", "dest"])["load"].sum()
    S3_IP, MY_IP = data_stats.index[data_stats.argmax()]

    data_stats.to_csv(path / "network_stats.csv")

    # %%
    # df = df_raw[df_raw["src"] == S3_IP] # TODO since IP changes, we look at all incoming traffic
    df = df_raw[df_raw["dest"] == MY_IP][["load", "time"]].copy()
    df = df.reset_index(drop=True)

    # %%
    # df_udp = pd.DataFrame(udp_marks)
    df_udp = df_raw[df_raw["protocol"].str.contains("UDP")]
    df_udp = df_udp[df_udp["dest"] == "8.8.8.8"]
    df_udp = df_udp.reset_index(drop=True)

    payloads = df_udp["payload"]
    payloads = [bytes.fromhex(x).decode("utf-8").split("-") for x in payloads]

    df_udp["epoch"], df_udp["iter"], df_udp["is_start"] = zip(*payloads)
    df_udp[["epoch", "iter"]] = df_udp[["epoch", "iter"]].apply(pd.to_numeric)
    df_udp = df_udp.drop(["src", "dest", "load", "protocol", "payload"], axis=1)

    # %%
    i = 0
    udp_times = df_udp["time"].to_numpy()
    iters_to_packets = defaultdict(int)
    for row in df.itertuples():
        time_cur = udp_times[i]
        time_next = udp_times[i + 1]

        if row.time < time_cur:
            continue
        else:

            break_loop = False
            while row.time > time_next:
                i += 1
                if i + 1 < udp_times.shape[0]:
                    time_cur = udp_times[i]
                    time_next = udp_times[i + 1]
                else:
                    break_loop = True
                    break

            if break_loop:
                break

            iters_to_packets[i] += row.load

    # %%
    df_u2 = df_udp.copy()
    df_u2["load"] = df_u2.index.map(iters_to_packets)
    df_u2 = pd.pivot(
        df_u2, index=["epoch", "iter"], columns=["is_start"], values=["load"]
    ).reset_index()
    df_u2.columns = ["epoch", "iter", "l_between", "l_inter"]

    # %%
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bins = 10

    ax = axs[0]
    s = df_u2["l_inter"]
    M = int(s.max() + 10 - (s.max() % 10))
    bins_0 = range(0, M, M // bins)
    ax.set_xticks(bins_0)
    s.hist(bins=bins_0, ax=ax)
    ax.set_title("Data during iterations")

    ax = axs[1]
    s = df_u2["l_between"]
    M_1 = int(s.max() + 10 - (s.max() % 10))
    bins_1 = range(0, M_1, M_1 // bins)
    ax.set_xticks(bins_1)
    s.hist(bins=bins_1, ax=ax)
    ax.set_title("Data between iterations")

    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("Transfered Data [MB]")
        ax.set_ylabel("# Iterations")

    fig.tight_layout()
    fig.savefig(path / "network_usage_during_versus_inter.jpg")
    # %%

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    M = int(df_u2[["l_between", "l_inter"]].max().max() * 0.8)

    ax = axs[0]
    s = df_u2.groupby("iter")["l_inter"].mean()
    x = np.array(range(s.shape[0]))
    ax.bar(x, s)
    ax.set_title("Data during iterations")

    ax = axs[1]
    s = df_u2.groupby("iter")["l_between"].mean()
    x = np.array(range(s.shape[0]))
    ax.bar(x, s)
    ax.set_title("Data between iterations")
    ax.axhline(y=M, label=f"{M} MB", c="k", ls="--")

    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Transfered Data [MB]")
        ax.axhline(y=5, label="5MB", c="k", ls="-.")
        ax.axhline(y=1, label="1 MB", c="k", ls="-")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path / "network_usage_per_iteration.jpg")


# %%
