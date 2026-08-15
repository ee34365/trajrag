import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("takeout", "geolife")


def setup(dataset: str | None = None) -> str:
    dataset = (dataset or os.environ.get("TRAJRAG_DATASET", "takeout")).lower()
    if dataset not in DATASETS:
        raise ValueError(f"TRAJRAG_DATASET must be one of {DATASETS}, got {dataset!r}")
    core = os.path.join(_HERE, "trajrag", "core")
    ds_dir = os.path.join(_HERE, "trajrag", "datasets", dataset)
    for d in (core, ds_dir):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    os.environ["TRAJRAG_DATASET"] = dataset
    return dataset


def dataset_dir(dataset: str | None = None) -> str:
    dataset = (dataset or os.environ.get("TRAJRAG_DATASET", "takeout")).lower()
    return os.path.join(_HERE, "trajrag", "datasets", dataset)
