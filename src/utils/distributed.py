import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from contextlib import closing
import socket

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)


def get_worker_info():
    worker_info = (
        torch.utils.data.get_worker_info()
    )  # TODO: fix for the case in which it is none
    if worker_info is None:
        return 1, 0
    worker_id = worker_info.id
    total_workers = worker_info.num_workers
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    total_workers *= world_size
    global_worker_id = worker_id * world_size + rank  # is it rank or rank_id
    return total_workers, global_worker_id


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # changed from "nccl" to "gloo" due to issues with the former
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
