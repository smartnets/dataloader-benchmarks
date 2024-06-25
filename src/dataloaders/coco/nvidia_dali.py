# Adapted from: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/detection_pipeline.html
from src.dataloaders.base import DataLoader
from src.datasets.coco.index import CocoDatasets
from torchvision import transforms
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


DATASET = CocoDatasets["nvidia_dali"]


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


    @pipeline_def(enable_conditionals=True)
    def create_dali_pipeline(self, dali_cpu=False, is_training=True):

        if is_training:
            file_root = "datasets/coco/shared/train" # Hardcoded path
            annotations_file = "datasets/coco/shared/annotations/instances_train2017.json"
        else:
            file_root = "datasets/coco/shared/val"
            annotations_file = "datasets/shared/annotations/instances_val2017.json"
        
        images, bboxes, labels = fn.readers.coco(file_root=file_root,
                                                annotations_file=annotations_file,
                                                skip_empty=True,
                                                shard_id=self.rank,
                                                num_shards=self.world_size,
                                                random_shuffle=is_training,
                                                pad_last_batch=True,
                                                name="Reader",
                                                ratio=True)
    
        crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(bboxes, labels,
                                                                    device="cpu",
                                                                    aspect_ratio=[0.5, 2.0],
                                                                    thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                                                    scaling=[0.3, 1.0],
                                                                    bbox_layout="xyXY",
                                                                    allow_no_crop=True,
                                                                    num_attempts=50)
        
        images = fn.decoders.image_slice(images, crop_begin, crop_size, device="mixed", output_type=types.RGB)
        flip_coin = fn.random.coin_flip(probability=0.5)

        bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=flip_coin)
        images = fn.crop_mirror_normalize(images,
                                        crop=(64, 64),
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        mirror=flip_coin,
                                        dtype=types.FLOAT,
                                        output_layout="CHW",
                                        pad_output=False)

        scales = [64]
        ratios = [0.5, 1.0, 2.0]

        # Generate anchors in xyWH format
        anchors = []
        for scale in scales:
            for ratio in ratios:
                width = scale * (ratio ** 0.5)
                height = scale / (ratio ** 0.5)
                
                # Calculate x_center and y_center
                x_center = 0  # Assuming anchors are centered at (0, 0)
                y_center = 0
                
                # Append anchor in xyWH format
                anchors.append(x_center)
                anchors.append(y_center)
                anchors.append(width)
                anchors.append(height)
        
        bboxes, labels = fn.box_encoder(bboxes, labels,
                                        criteria=0.5,
                                        scale=1.0,
                                        anchors=anchors)
        images = images.gpu()
        labels = labels.gpu()
        bboxes = bboxes.gpu()
        return images, bboxes, labels
    

    def get_train_loader(self, **kwargs):
        train_pipe = self.create_dali_pipeline(batch_size=kwargs["batch_size"],
                                          num_threads=kwargs["num_workers"],
                                          device_id=self.rank,
                                          is_training=True)
        train_pipe.build()
        train_loader = DALIGenericIterator(train_pipe,  ["images", "boxes", "labels"], reader_name="Reader",
                                                  last_batch_policy=LastBatchPolicy.PARTIAL,
                                                  auto_reset=True)
        return train_loader
    
    def get_val_loader(self, **kwargs):
        val_pipe = self.create_dali_pipeline(batch_size=kwargs["batch_size"],
                                          num_threads=kwargs["num_workers"],
                                          device_id=self.rank,
                                          is_training=False)
        val_pipe.build()
        val_loader = DALIGenericIterator(val_pipe, ["data", "label"], reader_name="Reader",
                                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                                auto_reset=True)
        return val_loader
    
    def get_test_loader(self, **kwargs):
        test_pipe = self.create_dali_pipeline(batch_size=kwargs["batch_size"],
                                          num_threads=kwargs["num_workers"],
                                          device_id=self.rank,
                                          is_training=False)
        test_pipe.build()
        test_loader = DALIGenericIterator(test_pipe, ["data", "label"], reader_name="Reader",
                                                 last_batch_policy=LastBatchPolicy.PARTIAL,
                                                 auto_reset=True)
        return test_loader