import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



from new_metrics.ImageClassificationMetric import ImageClassificationMetric
import numpy as np
import tensorflow as tf
from loaders.CIFAR10.loaderCIFAR10 import load
from tqdm.auto import tqdm 
from typing import Callable

dataset = load()
network  = tf.keras.models.load_model(
    "/Users/domenicopalumbo/keras_weight_injector/models/CIFAR10/ResNet20.keras"
    )


batch = 32
#dataset = dataset.take(6) # take 2 elements to reduce computation time
batched = dataset.batch(batch)

pbar = tqdm(
        batched,
        leave=False, 
        desc = "Dataset Inference",
        colour = "green",
    )


batch_predictions = []
batch_labels = []
for x in pbar:
    data, label = x
    out = network(data)
    batch_predictions.append(out)
    batch_labels.append(label)

predictions = tf.concat(batch_predictions, axis=0)
labels = tf.concat(batch_labels, axis=0)


#metric = ImageClassificationMetric(predictions, None, labels, num_classes=21)
#clean_metric = metric.clean_metric() # (dataset, 21)
#print("Clean metric: ", clean_metric)

#------------------------------------------------------------------------------#
def make_k_accuracy(k: int) -> Callable:
    def k_accuracy(scores: np.ndarray, labels: np.ndarray) -> int:
        top_k = scores.argpartition(-k, axis=-1)[:, -k:]
        print("Top k shape: ", top_k.shape)
        print("Labels shape: ", labels.shape)
        return (top_k == labels).any(axis=1).sum()

    return k_accuracy

def make_k_robustness(k: int, golden_labels: np.ndarray) -> Callable:
    top_k = make_k_accuracy(k)

    def k_robustness(scores: np.ndarray) -> int:
        return top_k(scores, golden_labels)

    return k_robustness


accuracy_5 = make_k_accuracy(5)
metric = accuracy_5(predictions.numpy(), labels.numpy())


clean_labels = predictions.numpy().argmax(axis=-1).reshape(-1, 1) #.astype(np.uint8)
robustness_5 = make_k_robustness(5, clean_labels)

metric = robustness_5(predictions.numpy())

print("Labels dtype: ", labels.numpy().dtype)
print("Clean labels dtype: ", clean_labels.dtype)
print("predictions shape: ", predictions.shape)
print("Clean labels shape: ", clean_labels.shape)
print("metric: ", metric)
