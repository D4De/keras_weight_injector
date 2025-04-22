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
