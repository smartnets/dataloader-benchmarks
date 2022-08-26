import time
import json
import os
import torch
import datetime
import numpy as np
from collections import defaultdict
from pathlib import Path
from src.profiling.networking import ping_udp
from src.profiling.pytorch import get_profiler
from src.profiling.metrics import (
    log_tcpdump,
    log_cpu,
    log_gpu,
)
from src.analysis.extract_metrics import collect_metrics
from src.config import settings as st
from copy import deepcopy


class MetricLogger(object):
    def __init__(self, run_id, rank):

        self.timer_epochs = {}
        self.timer_loader = defaultdict(list)
        self.losses = defaultdict(list)
        self.metrics = defaultdict(list)
        self.runtime = []
        self.proc_samples = defaultdict(int)
        self.res_dicts = {}
        self.completed_epochs = 0
        self.rank = rank
        self.run_id = run_id

        self._get_path()

    def _get_path(self):

        self.path = Path(st.local_results_dir)
        self.path /= st.dataset
        self.path /= st.library
        self.path /= str(self.run_id)

        self.path.mkdir(parents=True, exist_ok=True)

    def start_side_collectors(self):

        pid = os.getpid()
        # p1 = log_nvidia_smi(self.path)
        # p2 = log_nvidia_dmon(self.path)
        if self.rank == 0:
            p3 = log_tcpdump(self.path)
            p4 = log_cpu(self.path, pid)
            p5 = log_gpu(self.path)
            self.pids = [p3, p4, p5]

        self.timer_epochs["start"] = time.perf_counter()

    def end_side_collectors(self):
        self.timer_epochs["end"] = time.perf_counter() - self.timer_epochs["start"]
        if self.rank == 0:
            [p.terminate() for p in self.pids]

    def log_loss(self, val, mode):
        self.losses[mode].append(val)

    def log_metric(self, val, mode):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.metrics[mode].append(val)

    def accumulate_samples(self, epoch, N):
        self.proc_samples[epoch] += N

    def log_time_loaders(self, mode):
        self.timer_loader[mode].append(time.perf_counter())

    def log_round(self, iterable, epoch, log_every: int = 20, **kwargs):

        # absolute_start = self.timer_epochs["start"]
        absolute_start = time.perf_counter()
        epoch_times = {"begin": time.perf_counter() - absolute_start}
        run_profile = epoch in kwargs["profiling_epochs"]
        rank = kwargs.get("rank", 0)

        cutoff = kwargs.get("cutoff")
        if run_profile:
            profiler = get_profiler(self.path, **kwargs)
            profiler.start()

        i = 0
        for it, obj in enumerate(iterable):
            # print(it, obj[0].shape)

            now = time.perf_counter() - absolute_start
            elapsed = now - epoch_times["begin"]
            self.runtime.append(elapsed)
            if cutoff > 0 and i >= cutoff:  # elapsed >= cutoff:
                # self.runtime = elapsed
                return

            if i % log_every == 0 and rank == 0:
                print(
                    f"Iter: {i}, Epoch: {epoch}, Elapsed: {elapsed:.2f}",
                    end="\r",
                    flush=True,
                )

            iter_time_start = time.perf_counter()
            ping_udp(epoch, i, "start")

            yield it, obj

            if run_profile:
                profiler.step()
            ping_udp(epoch, i, "end")
            iter_time_end = time.perf_counter()

            epoch_times[i] = {
                "start": iter_time_start - absolute_start,
                "end": iter_time_end - absolute_start,
            }
            i += 1

        if run_profile:
            profiler.stop()
        epoch_times["end"] = time.perf_counter() - absolute_start

        self.timer_epochs[epoch] = epoch_times
        self.completed_epochs += 1
        print("\n")

    def _save_loss(self):
        with open(self.path / "loss.txt", "w") as fh:

            columns = [
                self.losses["train"],
                self.losses["val"],
            ]

            if len(self.metrics["val"]) > 0:
                metric_keys = list(self.metrics["val"][0].keys())
                for key in metric_keys:
                    l = [x[key] for x in self.metrics["val"]]
                    columns.append(l)

            for line in zip(*columns):
                line = ",".join(map(str, line)) + "\n"
                fh.write(line)

    def _save_params(self):

        final_dict = self.get_final_dict()
        print(f"Samples per second: {final_dict['proc_samples_per_sec']:.2f}")
        with open(self.path / "parameters.json", "w") as fh:
            json.dump(final_dict, fh, indent=2)

        # if st.ENV_FOR_DYNACONF == "prod":
        #     upload_results(final_dict)

    def get_results_dict(self):
        """
        Logs benchmark model's input parameters in a
        .json file
        """
        k = []
        for i in range(self.completed_epochs):
            if i == st.num_epochs - 1:
                k.append(self.timer_epochs["end"] - self.timer_epochs[i]["begin"])
            else:
                k.append(
                    self.timer_epochs[i + 1]["begin"] - self.timer_epochs[i]["begin"]
                )

        # Here is where we calculate the speed
        # We are taking the time per batch and calculating the speed per batch
        # Then, we are taking the average. The first batch is considerably slower
        runtime = np.diff(self.runtime)
        speed = np.median(st.batch_size / runtime)

        res_dict = {
            "dataloader_start_time": self.timer_loader,
            "date": datetime.datetime.now().isoformat(),
            "all_runtimes": list(x for x in runtime),
            "runtime": self.runtime[-1],
            "proc_samples": self.proc_samples,
            "samples_per_sec": speed,
            "test_loss": self.losses.get("test", "nan"),
            "test_metric": self.metrics.get("test"),
            "total_training_time": self.timer_epochs["end"],
            "avg_time_per_epoch": sum(k) / len(k) if len(k) > 0 else None,
        }

        return res_dict

    def update_final_dicts(self, final_dicts: list):

        total_proc_samples = 0
        max_runtime = 0
        speed = 0
        for rank, res_dict in enumerate(final_dicts):
            if res_dict is None:
                continue
            self.res_dicts[rank] = res_dict

            speed += res_dict["samples_per_sec"]
            total_proc_samples += res_dict["proc_samples"][0]
            max_runtime = max([max_runtime, res_dict["runtime"]])

        self.res_dicts["proc_samples"] = total_proc_samples
        self.res_dicts["max_runtime"] = max_runtime
        self.res_dicts["proc_samples_per_sec"] = speed

    def get_final_dict(self):
        other_metrics = collect_metrics(self.path)
        final_dict = {
            "running_id": self.run_id,
            **self.res_dicts,
            **other_metrics,
            **deepcopy(st.as_dict()),
        }
        return final_dict

    def persist_metrics(self):

        self._save_loss()
        self._save_params()
