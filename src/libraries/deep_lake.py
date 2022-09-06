import sys
import numpy as np


def filter_by_class(dataset, active_class_numbers):

    query = "SELECT * WHERE "
    for cls in active_class_numbers:
        query += f"CONTAINS(labels, {cls}) OR "

    query = query[:-3]
    print(query)
    dataset = dataset.query(query)
    return dataset


def create_dataset(data, store, class_names, dataset_kind="random"):

    with store:
        # Create the tensors with names of your choice.
        store.create_tensor("images", htype="image", sample_compression="jpeg")
        store.create_tensor("labels", htype="class_label", class_names=class_names)
        if dataset_kind == "coco":
            store.create_tensor("boxes", htype="bbox")
            store.boxes.info.update(coords={"type": "fractional", "mode": "LTWH"})

        # Add arbitrary metadata - Optional
        store.info.update(description=dataset_kind)
    with store:
        for i, data in enumerate(data):
            image, label = data
            if dataset_kind == "coco":
                if len(label["categories"]) > 0:
                    store.images.append(image)
                    store.labels.append(label["categories"])
                    store.boxes.append(label["boxes"])
            else:
                store.append({"images": np.array(image), "labels": np.uint8(label)})

            print(f"Iteration {i:4d}", end="\r", flush=True, file=sys.stderr)

    return store
