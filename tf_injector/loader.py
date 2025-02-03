from tf_injector.utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    SUPPORTED_DATASETS,
    download_gtsrb,
)
from tf_injector.preprocessing import preprocessors, np_preprocessors

import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore

import os

import numpy as np
import tqdm


def load_gtsrb():
    dt_path = DEFAULT_DATASET_PATH / "GTSRB/GTSRB_keras"
    if not os.path.exists(dt_path):
        download_gtsrb()
    return tf.data.Dataset.load(str(dt_path), compression="GZIP")


def from_tensor_slices(loader):
    def wrapper():
        return tf.data.Dataset.from_tensor_slices(loader()[1])

    return wrapper


@from_tensor_slices
def load_cifar10():
    return keras.datasets.cifar10.load_data()


@from_tensor_slices
def load_cifar100():
    return keras.datasets.cifar100.load_data()


loaders = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "gtsrb": load_gtsrb,
}


def loader(dataset_name: str):
    return loaders[dataset_name]()

# testing helpers
def test_dt_distance(d1, d2):
    assert d1.shape == d2.shape, f"shapes differ: {d1.shape, d2.shape}"
    for dt1_l, dt2_l in tqdm.tqdm(zip(d1, d2), "running dataset test"):
        dt1 = dt1_l
        dt2 = dt2_l
        diff = dt1 - dt2
        assert not diff.any(), f"{diff[diff!=0]}"
    assert (d1 == d2).all()


def dt_to_np(dt):
    batches = []
    for batch, labels in dt:
        batch = batch.numpy()
        batches.append(batch)
    return np.stack(batches, axis=0)


def load_network(
    network_name: str,
    dataset_name: str,
    model_path=DEFAULT_MODEL_PATH,
    use_tf: bool = False
) -> tuple[keras.Model, tf.data.Dataset]:
    model_path = os.path.join(model_path, dataset_name, network_name + ".keras")
    print("loading model ", network_name)
    model = keras.models.load_model(model_path)
    print("done")
    assert dataset_name in SUPPORTED_DATASETS
    print("loading dataset...")
    d_load = loader(dataset_name.lower())
    print("loaded")
    if use_tf:
        dataset = d_load.map(preprocessors[dataset_name])
    else:
        dataset = d_load.map(np_preprocessors[dataset_name])

    return model, dataset
