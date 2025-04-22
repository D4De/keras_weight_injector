from tf_injector.new_metrics.metric import Metric
from typing import Callable
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


class ImageClassificationMetric(Metric):
    def __init__(self, clean_scores: tf.Tensor, clean_labels: tf.Tensor, labels: tf.Tensor):
        # TODO reimplement metrics using tf.Tensor
        super().__init__(clean_scores.numpy(), clean_labels.numpy(), labels.numpy())
        self.top_1_robustness = make_k_robustness(1, self.clean_labels)
        self.top_5_robustness = make_k_robustness(5, self.clean_labels)
        self.masked_counter = make_masked_counter(self.clean_scores)


    def clean_metric(self) -> tuple[int, ...]:
        return top_1_accuracy(self.clean_scores, self.labels), top_5_accuracy(self.clean_scores, self.labels)

    def clean_output(self) -> tuple:
        metric = self.clean_metric()
        padding = [None]
        return (*metric, *(padding*4))

    def faulty_output(self, faulty_scores: tf.Tensor, with_respect_to_labels:bool=False) -> tuple[int, ...]:
        faulty_scores = faulty_scores.numpy()
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
