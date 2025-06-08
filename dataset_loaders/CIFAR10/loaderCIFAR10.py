'''
Loading file for CIFAR10
'''
from tensorflow import keras
import tensorflow as tf
import numpy as np

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


#=================Preprocessing=========================#

def np_preprocessed_dataset():
    dataset = load_cifar10()
    
    # convert to numpy
    data_list = [(img, label) for img, label in dataset.as_numpy_iterator()]
    
    # preprocess images
    processed_images = []
    processed_labels = []
    
    for image, label in data_list:
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

        image = np.array(image, dtype=np.float32)
        image = image / np.float32(255.0)
        image = (image - mean) / std
        
        label = np.array(label, np.uint8)

        processed_images.append(image)
        processed_labels.append(label)


    return tf.data.Dataset.from_tensor_slices(
        (np.array(processed_images), np.array(processed_labels))
    )


def tf_preprocessed_dataset():
    dataset = load_cifar10()
    

    def preprocess_image(image, label):


        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)


        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image - mean) / std
        
        
        return image, label
    
    return dataset.map(preprocess_image)



# to be loaded 
def load(DATASET_PATH : str):
    #return np_preprocessed_dataset()
    return tf_preprocessed_dataset()