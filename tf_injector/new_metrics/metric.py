from abc import abstractmethod, ABCMeta
import tensorflow as tf

class Metric(metaclass=ABCMeta):
    """
    Abstract class representing a set of metric functions.
    The class is initialised with data related to the clean inference,
    that can be used for the faulty metrics.
    This class implements two types of functions:
    - Metric functions that return a tuple containing the metrics
    - Output functions that return a tuple that will be used to fill the report file
    """
    def __init__(self, clean_scores : tf.Tensor, labels: tf.Tensor, num_classes: int):
        """
        Initialises the class with data related to the clean inference and the labels
        Args:
            clean_scores: the result of the evaluation of the dataset by the model
            clean_labels: the predicted labels in the clean run
            labels: ground-truth labels of the dataset
        """
        self.clean_scores = clean_scores
        self.labels = labels
        self.num_classes = num_classes
        self.clean_labels = self._evaluate_clean_labels(clean_scores) if clean_scores is not None else None

    def _evaluate_clean_labels(self, clean_scores: tf.Tensor) -> tf.Tensor:
        clean_scores = tf.argmax(clean_scores, axis=-1)
        clean_scores = tf.cast(clean_scores, tf.int64)
        return clean_scores

    @abstractmethod
    def clean_output(self) -> tuple:
        """
        This function uses the information retained in the class (clean scores and labels)
        to compute the metrics related to the clean run.

        This function will be called by the injector to obtain the values that will
        be written in the report file as the golden row (e.g. add commas if the metrics are 
        less than the faulty ones).
        For instance, if the clean metrics are TOP-1 and TOP-5 accuracy, but there
        are three more faulty metrics, the output would be like:
        (0.9182, 0.9954, None, None, None)
        Where None is used as placeholder for an empty cell in the CSV.
        """
        pass

    @abstractmethod
    def faulty_output(self, faulty_scores: tf.Tensor, with_respect_to_labels: bool = False) -> tuple:
        """
        This function will be called by the injector and provides the values that will
        be written in the report files in the faulty rows.
        """
        pass

    @abstractmethod
    def get_header(self) -> tuple[str]:
        """
        This function will be called by the injector to obtain the header of the CSV file.
        The header is a list of strings that will be used as the first row of the CSV file.
        """
        pass
