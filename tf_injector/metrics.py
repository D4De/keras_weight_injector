from typing import Callable
from abc import abstractmethod, ABCMeta
import numpy as np
import tensorflow as tf 


def make_k_accuracy(k: int) -> Callable:
    def k_accuracy(scores: np.ndarray, labels: np.ndarray) -> int:
        top_k = scores.argpartition(-k, axis=-1)[:, -k:]
        return (top_k == labels).any(axis=1).sum()

    return k_accuracy


def make_k_robustness(k: int, golden_labels: np.ndarray) -> Callable:
    top_k = make_k_accuracy(k)

    def k_robustness(scores: np.ndarray) -> int:
        return top_k(scores, golden_labels)

    return k_robustness


def make_masked_counter(golden_scores: np.ndarray) -> Callable:
    def masked_counter(faulty_scores: np.ndarray) -> int:
        return (faulty_scores == golden_scores).all(axis=1).sum()

    return masked_counter


def non_critical_counter(top_1_robust: int, masked_count: int) -> int:
    return top_1_robust - masked_count


def critical_counter(num_inferences: int, top_1_robust: int) -> int:
    return num_inferences - top_1_robust

top_1_accuracy = make_k_accuracy(1)
top_5_accuracy = make_k_accuracy(5)

class Metric(metaclass=ABCMeta):
    """
    Abstract class representing a set of metric functions.
    The class is initialised with data related to the clean inference,
    that can be used for the faulty metrics.
    This class implements two types of functions:
    - Metric functions that return a tuple containing the metrics
    - Output functions that return a tuple that will be used to fill the report file
    """
    def __init__(self, clean_scores: tf.Tensor, clean_labels: tf.Tensor, labels: tf.Tensor):
        """
        Initialises the class with data related to the clean inference and the labels
        Args:
            clean_scores: the result of the evaluation of the dataset by the model
            clean_labels: the predicted labels in the clean run
            labels: ground-truth labels of the dataset
        """
        self.clean_scores = clean_scores
        self.clean_labels = clean_labels
        self.labels = labels

    @abstractmethod
    def clean_metric(self) -> tuple[int, ...]:
        """
        This function uses the information retained in the class (clean scores and labels)
        to compute the metrics related to the clean run.
        """
        pass

    @abstractmethod
    def clean_output(self) -> tuple:
        """
        This function will be called by the injector to obtain the values that will
        be written in the report file as the golden row (e.g. add commas if the metrics are 
        less than the faulty ones)
        """
        pass

    @abstractmethod
    def faulty_output(self, faulty_scores: np.ndarray) -> tuple:
        """
        This function will be called by the injector and provides the values that will
        be written in the report files in the faulty rows.
        """
        pass


class ImageClassificationMetric(Metric):
    def __init__(self, clean_scores: np.ndarray, clean_labels: np.ndarray, labels: np.ndarray):
        super().__init__(clean_scores, clean_labels, labels)
        self.top_1_robustness = make_k_robustness(1, clean_labels)
        self.top_5_robustness = make_k_robustness(5, clean_labels)
        self.masked_counter = make_masked_counter(clean_scores)


    def clean_metric(self) -> tuple[int, ...]:
        return top_1_accuracy(self.clean_scores, self.labels), top_5_accuracy(self.clean_scores, self.labels)

    def clean_output(self) -> tuple:
        metric = self.clean_metric()
        padding = [None]
        return (*metric, *(padding*4))

    def faulty_output(self, faulty_scores: np.ndarray) -> tuple[int, ...]:
        top_1_robust = self.top_1_robustness(faulty_scores)
        masked_count = self.masked_counter(faulty_scores)

        return (
            top_1_accuracy(faulty_scores, self.labels),
            top_5_accuracy(faulty_scores, self.labels),
            top_1_robust,
            self.top_5_robustness(faulty_scores),
            masked_count,
            non_critical_counter(top_1_robust, masked_count),
            critical_counter(len(self.clean_labels), top_1_robust),
        )

def compute_IOU(x1, x2, numclass):
    x1 = tf.argmax(x1, axis=-1, output_type=tf.dtypes.uint16)
    x2 = tf.argmax(x2, axis=-1, output_type=tf.dtypes.uint16)
    ious = []
    for cls in range(numclass):
        clsx1 = x1 == cls
        clsx2 = x2 == cls
        inter = tf.reduce_sum(
            tf.cast(tf.logical_and(clsx1, clsx2), tf.uint64),
            axis=[1,2]
        )
        union = tf.reduce_sum( 
            tf.cast(tf.logical_or(clsx1, clsx2),tf.uint64),
            axis=[1,2]
        )
        iou = inter / union
        ious.append(iou)
    ious = tf.transpose(tf.stack(ious))
    return ious

class ImageIntersectionOverUnionMetric(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, clean_scores, clean_labels, labels):
        super().__init__(clean_scores, clean_labels, labels)
        return self
       
    def clean_metric(self):
        return compute_IOU(self.clean_scores, self.labels)
        
    def clean_output(self):
       return [1] * self.num_classes

    def faulty_output(self, faulty_scores):
        return compute_IOU(
            self.clean_scores,
            faulty_scores,
            self.num_classes
        )

def compute_pixel_accuracy(x1, x2):
    x1 = tf.argmax(x1, axis=-1, output_type=tf.dtypes.uint16)
    x2 = tf.argmax(x2, axis=-1, output_type=tf.dtypes.uint16)
    diff = x1 == x2
    totalPixels = x1.shape[0] * tf.math.reduce_prod( x1.shape[1:] )
    correctPixels = tf.math.reduce_sum( 
        tf.cast( diff, tf.int64 ),
        axis = [1,2]
    )
    return correctPixels

class PixelAccuracyMetric(Metric):
    def __init__(self, clean_scores, clean_labels, labels):
        super().__init__(clean_scores, clean_labels, labels)
       
    def clean_metric(self):
        return compute_pixel_accuracy(self.clean_scores, self.labels)
        
    def clean_output(self):
       return [1]

    def faulty_output(self, faulty_scores):
        return compute_pixel_accuracy(
            self.clean_scores,
            faulty_scores,
        )
