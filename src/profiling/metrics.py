import subprocess
import time
import requests
from pathlib import Path


def log_nvidia_smi(path: Path):

    if isinstance(path, str):
        path = Path(path)

    NVIDIA_SMI_ARGS = [
        "/usr/bin/nvidia-smi",
        "--query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
        "--format=csv",
        "--loop=1",
        "-f",
        path / "nvidia-smi.log",
    ]

    pid = subprocess.Popen(NVIDIA_SMI_ARGS)
    return pid


def log_cpu(path: Path, pid: int):

    if isinstance(path, str):
        path = Path(path)

    ARGS = [
        "/home/worker/workspace/src/profiling/cpu_usage.sh",
        str(pid),
        str(path / "cpu_usage.txt"),
    ]
    pid = subprocess.Popen(ARGS)
    return pid


def log_gpu(path: Path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = [
        "/home/worker/workspace/src/profiling/gpu_usage.sh",
        str(path / "gpu_usage.txt"),
    ]
    pid = subprocess.Popen(ARGS)
    return pid


def log_nvidia_dmon(path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = ["nvidia-smi", "dmon", "-o", "DT", "-f", path / "nvidia-dmon.log"]

    pid = subprocess.Popen(ARGS)
    return pid


def log_tcpdump(path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = ["tcpdump", "-s", "96", "udp", "or", "tcp", "-w", path / "dump.pcap"]
    pid = subprocess.Popen(ARGS)
    return pid


if __name__ == "__main__":
    pass
    # import os
    # print(os.getpid())
    # while 1:
    #     time.sleep(1)
    # p_nsmi = log_nvidia_smi("nvidia.log")
    # p_ndmn = log_nvidia_dmon("nvidia_dmon.log")
    # p_tdump = log_tcpdump("tcp")

    # for i in range(10):
    #     print(i)
    #     requests.get(f"http://www.google.com/q={i}")
    #     time.sleep(1)

    # p_nsmi.terminate()
    # p_ndmn.terminate()
    # p_tdump.terminate()
