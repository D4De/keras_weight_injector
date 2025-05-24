'''
Loading file for CIFAR100
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
def load_cifar100():
    return keras.datasets.cifar100.load_data()


#=================Preprocessing=========================#

def np_preprocessed_dataset():
    dataset = load_cifar100()
    
    # convert to numpy
    data_list = [(img, label) for img, label in dataset.as_numpy_iterator()]
    
    # preprocess images
    processed_images = []
    processed_labels = []
    
    for image, label in data_list:

        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=np.float32)
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=np.float32)

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
    dataset = load_cifar100()
    
    def preprocess_image(image, label):

        # Standardizzazione con media e deviazione standard del dataset CIFAR-10
        mean = tf.constant([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=tf.float32)
        std = tf.constant([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=tf.float32)

        # Cast to float32 and normalize [0-255] -> [0-1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image - mean) / std
        
        
        return image, label
    
    # Preprocess the whole dataset
    return dataset.map(preprocess_image)



# to be loaded 
def load():
    return np_preprocessed_dataset()
    #tf_preprocessed_dataset()
