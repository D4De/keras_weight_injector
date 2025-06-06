import tensorflow as tf


def func1():
    """
    Other functions can be defined in this file 
    but remember to set the principal function in 
    --function_name  arg
    """
    pass


def load() -> tf.data.Dataset:
    """
    - Load the dataset from an URL or a local path and 
    - Parse it into a TensorFlow dataset.
    - Returns the dataset with no bathces and organized in (input, label) pairs.
    """
    return dataset

