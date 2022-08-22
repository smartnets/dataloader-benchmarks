import sys
import numpy as np
from src.datasets.base import Dataset
from squirrel.serialization import MessagepackSerializer
from squirrel.store import SquirrelStore
from src.datasets.cifar10.base import LABELS_DICT, get_cifar10
from src.libraries.squirrel import build_dataset


class SquirrelDataset(Dataset):
    def __init__(self):
        super().__init__("cifar10", "squirrel")
        self.filtering = None

        self.num_shards = {  # These are manually hardcoded after looking at how many shards are created
            "train": 10,
            "test": 10,
            "val": 10,
        }

    def _generate(self, store, dataset, mode):

        N_SHARDS = self.num_shards[mode]
        samples = []
        for index, (input, output) in enumerate(dataset):
            samples.append({
                "__key__": "sample%06d" % index,
                "jpg": np.array(input),
                "cls": output,
            })
        shards = [samples[i::N_SHARDS] for i in range(N_SHARDS)]

        for index, shard in enumerate(shards):
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
            store.set(
                shard,
                key=f"shard_%d" % index,
            )

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True)

        # Initialization of SquirrelStore with local path
        store = SquirrelStore(
            url=str(path), serializer=MessagepackSerializer())
        cifar = get_cifar10(mode, download=True, transform=transforms)

        self._generate(store, cifar, mode)

        # Sharding

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None

        # Initialization of SquirrelStore with local path
        store = SquirrelStore(
            url=str(path), serializer=MessagepackSerializer())
        cifar = get_cifar10(mode, download=True, transform=transforms)

        self._generate(store, cifar, mode)

    def get_local(self, mode="train", transforms=None, filtering=False, filtering_classes=[], distributed=False, batch_size=None):
        path = self.get_local_path()
        path /= f"{mode}"

        if filtering:
            FC = [LABELS_DICT[i] for i in filtering_classes]
        else:
            FC = []
        

        return build_dataset(
            path,
            transforms,
            distributed,
            batch_size,
            filtering,
            FC
        )

    def get_remote(self, mode="train", transforms=None, filtering=False, filtering_classes=[], distributed=False, batch_size=None):
        path = self.get_local_path()
        path /= f"{mode}"

        if filtering:
            FC = [LABELS_DICT[i] for i in filtering_classes]
        else:
            FC = []

        return build_dataset(
            path,
            transforms,
            distributed,
            batch_size,
            filtering,
            FC
        )