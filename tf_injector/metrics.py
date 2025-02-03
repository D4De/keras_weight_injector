from typing import Callable
import numpy as np


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


def gold_row_std_metric(scores: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    return top_1_accuracy(scores, labels), top_5_accuracy(scores, labels)


def make_faulty_row_std_metric(
    golden_scores: np.ndarray, golden_labels: np.ndarray
) -> Callable:
    top_1_robustness = make_k_robustness(1, golden_labels)
    top_5_robustness = make_k_robustness(5, golden_labels)
    masked_counter = make_masked_counter(golden_scores)

    def faulty_row_std_metric(
        faulty_scores: np.ndarray, labels: np.ndarray
    ) -> tuple[int, int, int, int, int, int, int]:
        top_1_robust = top_1_robustness(faulty_scores)
        masked_count = masked_counter(faulty_scores)

        return (
            top_1_accuracy(faulty_scores, labels),
            top_5_accuracy(faulty_scores, labels),
            top_1_robust,
            top_5_robustness(faulty_scores),
            masked_count,
            non_critical_counter(top_1_robust, masked_count),
            critical_counter(len(golden_labels), top_1_robust),
        )

    return faulty_row_std_metric
