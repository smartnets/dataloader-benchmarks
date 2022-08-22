import sys
from src.libraries.webdataset import build_dataset
from src.utils.persist import save_results_s3
import webdataset as wds
from src.datasets.base import Dataset
from src.datasets.cifar10.base import get_cifar10, LABELS_DICT

# def identity(x): return x

class WebdatasetDataset(Dataset):
    def __init__(self):
        super().__init__("cifar10", "webdataset")
        self.filtering = None


        self.shard_prefix = "shard-%06d.tar"
        self.num_shards = {  # These are manually hardcoded after looking at how many shards are created
            "train": 1,
            "test": 1,
            "val": 1,
        }


    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True)

        sink = wds.ShardWriter(str(path / self.shard_prefix))
        cifar = get_cifar10(mode, download=True, transform=transforms)

        for index, (input, output) in enumerate(cifar):
            if index % 1000 == 0:
                print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
            sink.write(
                {
                    "__key__": "sample%06d" % index,
                    "jpg": input,
                    "cls": output,
                }
            )
        sink.close()

    def _remote_name(self, shard, mode):

        return f"datasets/{self.dataset}/{self.library}/{mode}/{self.shard_prefix}" % shard

    def generate_remotely(self, mode="train", transforms=None):
        self.generate_locally(mode=mode, transforms=transforms)

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None

        for i in range(self.num_shards[mode]):
            name = self.shard_prefix % i

            local_path = self.get_local_path()
            local_path /= f"{mode}"
            local_path /= name

            remote_name = self._remote_name(i, mode)
            save_results_s3(str(local_path), remote_name)

    def get_local(self, mode="train", transforms=None, filtering=False, filtering_classes=[], batch_size=1, distributed=False):

        path = self.get_local_path()
        path /= f"{mode}"
        path /= "shard-{%06d..%06d}.tar" % (0, self.num_shards[mode] - 1)
        path = str(path)

        if filtering:
            FC = [LABELS_DICT[c] for c in filtering_classes]
        else:
            FC = []

        return build_dataset(
            path,
            distributed,
            transforms,
            filtering,
            FC,
            batch_size,
        )

    def get_remote(self, mode="train", transforms=None, filtering=False, filtering_classes=[], batch_size=1, distributed=False):

        path = self.get_remote_path()
        path += f"/{mode}"
        path += "/shard-{%06d..%06d}.tar -" % (0, self.num_shards[mode] - 1)
        path = "pipe:s3cmd -q get --force " + path

        if filtering:
            FC = [LABELS_DICT[c] for c in filtering_classes]
        else:
            FC = []
        
        return build_dataset(
            path,
            distributed,
            transforms,
            filtering,
            FC,
            batch_size,
        )
