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

    pid = subprocess.Popen(
        NVIDIA_SMI_ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    return pid


def log_cpu(path: Path, pid: int):

    if isinstance(path, str):
        path = Path(path)

    ARGS = [
        "/home/worker/workspace/src/profiling/cpu_usage.sh",
        str(pid),
        str(path / "cpu_usage.txt"),
    ]
    pid = subprocess.Popen(ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pid


def log_gpu(path: Path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = [
        "/home/worker/workspace/src/profiling/gpu_usage.sh",
        str(path / "gpu_usage.txt"),
    ]
    pid = subprocess.Popen(ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pid


def log_nvidia_dmon(path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = ["nvidia-smi", "dmon", "-o", "DT", "-f", path / "nvidia-dmon.log"]

    pid = subprocess.Popen(ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pid


def log_tcpdump(path):

    if isinstance(path, str):
        path = Path(path)

    ARGS = ["tcpdump", "-s", "96", "udp", "or", "tcp", "-w", path / "dump.pcap"]
    pid = subprocess.Popen(ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pid


if __name__ == "__main__":
    pass
