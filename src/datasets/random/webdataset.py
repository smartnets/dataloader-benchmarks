import sys
from src.utils.persist import save_results_s3
import webdataset as wds
from src.datasets.base import Dataset
from src.datasets.random.base import get_random, LABELS_DICT
from src.libraries.webdataset import build_dataset

def identity(x): return x

class WebdatasetDataset(Dataset):
    def __init__(self):
        super().__init__("random", "webdataset")

        self.shard_prefix = "shard-%06d.tar"
        self.num_shards = {  # These are manually hardcoded after looking at how many shards are created
            "train": 90,
            "test": 1,
            "val": 10,
        }

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True)

        sink = wds.ShardWriter(str(path / self.shard_prefix), maxcount=1000)
        random = get_random(mode, download=True, transform=transforms)

        for index, (input, output) in enumerate(random):
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

    def _build_dataset(self, path, transforms=None):

        dataset = (
            wds.WebDataset(path, nodesplitter=wds.shardlists.split_by_node,)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(transforms, identity)
        )
        return dataset

    def _build_filtered_dataset(self, path, transforms=None, filtering=False):

        def identity(x): return x

        print(f"Filtering for class names : {filtering}")
        labels_list = [LABELS_DICT[i] for i in filtering]
        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .select(lambda sample: sample["cls"] in labels_list)
            .to_tuple("jpg", "cls")
            .map_tuple(transforms, identity)
        )
        return dataset

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
