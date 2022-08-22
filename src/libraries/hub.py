import sys
import numpy as np


def create_dataset(dataset, store, class_names):

    with store:
        # Create the tensors with names of your choice.
        store.create_tensor("images", htype="image", sample_compression="jpeg")
        store.create_tensor("labels", htype="class_label", class_names=class_names)

        # Add arbitrary metadata - Optional
        # store.info.update(description="Random")
        # store.images.info.update(camera_type="SLR")
    with store:
        for i, (image, label) in enumerate(dataset):
            print(f"Iteration {i:4d}", end="\r", flush=True, file=sys.stderr)
            store.append({"images": np.array(image), "labels": np.uint8(label)})

    return store

def filter_by_classs(dataset, classes):
    query_str = f"labels in {classes}"
    print(query_str)
    return dataset.filter(query_str)