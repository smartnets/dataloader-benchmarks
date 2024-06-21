# Adapted from: https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
from torchvision import transforms
from src.datasets.cifar10.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)

import torch.utils.data as torch_data
from src.libraries.pytorch import filter_by_class


import torch.utils.data
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


DATASET = CIFAR10Datasets["nvidia_dali"]


class NvidiaDaliLoader(DataLoader):
    def __init__(
        self,
        filtering: bool = False,
        filtering_classes: list = None,
        remote: bool = False,
        distributed: bool = False,
        world_size: int = None,
        rank: int = None,
    ):
        super().__init__(
            filtering=filtering,
            filtering_classes=filtering_classes,
            remote=remote,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
        )

        self.filtering = filtering
        self.filtering_classes = filtering_classes
        self.remote = remote
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank

        self.train_transforms = get_train_transforms()
        self.eval_transforms = get_eval_transforms()

    @pipeline_def
    def create_dali_pipeline(self, crop, size, dali_cpu=False, is_training=True):

        if is_training:
            file_root = DATASET.get_local_path() / "train"
        else:
            file_root = DATASET.get_local_path() / "val"

        images, labels = fn.readers.file(file_root=file_root,
                                        shard_id=self.rank,
                                        num_shards=self.world_size,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        
        if is_training:
            images = fn.decoders.image_random_crop(images,
                                                device=decoder_device, output_type=types.RGB,
                                                device_memory_padding=device_memory_padding,
                                                host_memory_padding=host_memory_padding,
                                                preallocate_width_hint=preallocate_width_hint,
                                                preallocate_height_hint=preallocate_height_hint,
                                                random_aspect_ratio=[0.8, 1.25],
                                                random_area=[0.1, 1.0],
                                                num_attempts=100)
            images = fn.resize(images,
                            device=dali_device,
                            resize_x=crop,
                            resize_y=crop,
                            interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images,
                                    device=decoder_device,
                                    output_type=types.RGB)
            images = fn.resize(images,
                            device=dali_device,
                            size=size,
                            mode="not_smaller",
                            interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(images.gpu(),
                                        dtype=types.FLOAT,
                                        output_layout="CHW",
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        mirror=mirror)
        labels = labels.gpu()
        return images, labels
    

    def get_train_loader(self, **kwargs):
        train_pipe = self.create_dali_pipeline(batch_size=kwargs["batch_size"],
                                          num_threads=kwargs["num_workers"],
                                          device_id=self.rank,
                                          crop=32,
                                          size=32,
                                          is_training=True)
        train_pipe.build()
        train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",
                                                  last_batch_policy=LastBatchPolicy.PARTIAL,
                                                  auto_reset=True)
        return train_loader