import sys
import numpy as np
from src.datasets.base import Dataset
from squirrel.serialization import MessagepackSerializer
from squirrel.store import SquirrelStore
from src.datasets.coco.base import LABEL_DICT, get_coco, get_size

from src.libraries.squirrel import build_dataset


class SquirrelDataset(Dataset):
    def __init__(self):
        super().__init__("coco", "squirrel")

        self.num_shards = {  # These are manually hardcoded after looking at how many shards are created
            "train": 50,
            "test": 10,
            "val": 10,
        }

    def _generate(self, store, dataset, mode):

        N_SHARDS = self.num_shards[mode]
        samples = []

        N = get_size(mode)
        size = N // N_SHARDS
        for i in range(N_SHARDS):
            samples = []
            print(f"Working on shard: {i}")
            for j in range(i * size, (i + 1) * size):
                print(f"{j:6d} / {size}", end="\r", flush=True, file=sys.stderr)
                input, output = dataset[j]
                samples.append({
                    "__key__": "sample%06d" % j,
                    "jpg": np.array(input),
                    "labels": output["categories"],
                    "boxes": output["boxes"],
                })
            store.set(
                samples,
                key=f"shard_%d" % i,
            )
            

            
        # for index, (input, output) in enumerate(dataset):
        #     samples.append({
        #         "__key__": "sample%06d" % index,
        #         "jpg": np.array(input),
        #         "labels": output["categories"],
        #         "boxes": output["boxes"],
        #     })
        # shards = [samples[i::N_SHARDS] for i in range(N_SHARDS)]

        # for index, shard in enumerate(shards):
        #     print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        #     store.set(
        #         shard,
        #         key=f"shard_%d" % index,
        #     )

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True)

        # Initialization of SquirrelStore with local path
        store = SquirrelStore(
            url=str(path), serializer=MessagepackSerializer())
        coco = get_coco(mode, None)

        self._generate(store, coco, mode)

        # Sharding

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None

        # Initialization of SquirrelStore with local path
        store = SquirrelStore(
            url=str(path), serializer=MessagepackSerializer())
        coco = get_coco(mode, None)

        self._generate(store, coco, mode)

    def get_local(self, mode="train", transforms=None, filtering=False, filtering_classes=[], distributed=False, batch_size=None):
        path = self.get_local_path()
        path /= f"{mode}"

        if filtering:
            FC = [LABEL_DICT[i] for i in filtering_classes]
        else:
            FC = []

        return build_dataset(
            path,
            transforms,
            distributed,
            batch_size,
            filtering,
            FC,
            "coco",
        )

    def get_remote(self, mode="train", transforms=None, filtering=False, filtering_classes=[], distributed=False, batch_size=None):
        path = self.get_local_path()
        path /= f"{mode}"

        if filtering:
            FC = [LABEL_DICT[i] for i in filtering_classes]
        else:
            FC = []

        return build_dataset(
            path,
            transforms,
            distributed,
            batch_size,
            filtering,
            FC,
            "coco"
        )
