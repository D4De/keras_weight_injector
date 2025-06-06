from tf_injector.metrics.metric import Metric
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
    def __init__(self, clean_scores: tf.Tensor, labels: tf.Tensor, num_classes):
        clean_scores = clean_scores.numpy() if clean_scores is not None else None
        labels = labels.numpy() if labels is not None else None

        super().__init__(clean_scores, labels, num_classes)
        if clean_scores is not None:
            self.clean_labels = self._evaluate_clean_labels(clean_scores)
            self.num_classes = len(np.unique(self.clean_labels))

        self.top_1_robustness = make_k_robustness(1, self.clean_labels)
        self.top_5_robustness = make_k_robustness(5, self.clean_labels)
        self.masked_counter = make_masked_counter(self.clean_scores)

    def _evaluate_clean_labels(self, clean_score: np.ndarray) -> np.ndarray:
        """
        (data, num_classes) -> (label)
        """
        return np.argmax(clean_score, axis=-1).reshape(-1, 1) #.astype(np.uint8)

    def clean_output(self) -> tuple:
        metric = top_1_accuracy(self.clean_scores, self.labels), top_5_accuracy(self.clean_scores, self.labels)
        padding = [None]
        return (*metric, *(padding*5))

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
    
    def get_header(self) -> tuple[str, ...]:
        return (
            "top_1_correct",
            "top_5_correct",
            "top_1_robust",
            "top_5_robust",
            "masked",
            "non_critical",
            "critical"
        )
