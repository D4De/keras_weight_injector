import tensorflow as tf  # type:ignore
from typing import Callable

import numpy as np

TransformType = tuple[float, float, float]


def make_preprocessor(mean: TransformType, std: TransformType) -> Callable:
    def preprocessor(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image - mean) / std
        return image, label

    return preprocessor


def make_np_preprocessor(mean: TransformType, std: TransformType) -> Callable:
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)

    @tf.numpy_function(Tout=[tf.float32, tf.uint8])
    def preprocessor(
        image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image, dtype=np.float32)
        image = image / np.float32(255.0)
        image = (image - mean_np) / std_np
        return image, np.array(label, np.uint8)

    return preprocessor

# preprocessing data with stats from github.com/polito/PTInference
preprocessors = {
    "CIFAR10": make_preprocessor((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": make_preprocessor(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
    "GTSRB": make_preprocessor((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
}

np_preprocessors = {
    "CIFAR10": make_np_preprocessor((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": make_np_preprocessor(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
    "GTSRB": make_np_preprocessor((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
}
