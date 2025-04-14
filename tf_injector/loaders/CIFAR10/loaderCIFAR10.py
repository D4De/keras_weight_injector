'''
Loading file for CIFAR10
'''
from tensorflow import keras
import tensorflow as tf

def from_tensor_slices(loader):
    """
    Loader wrapper for keras datasets
    """
    def wrapper():
        return tf.data.Dataset.from_tensor_slices(loader()[1])

    return wrapper


@from_tensor_slices
def load_cifar10():
    return keras.datasets.cifar10.load_data()
