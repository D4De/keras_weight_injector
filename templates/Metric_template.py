from tf_injector.metrics.metric import Metric
import numpy as np
import tensorflow as tf

class Metric_template(Metric):
    def __init__(self, clean_scores, labels, num_classes):
        super().__init__(clean_scores, labels, num_classes)
        if clean_scores is not None:
            self.num_classes = clean_scores.shape[-1] 
            self.clean_labels = self.__evaluate_clean_labels(clean_scores) 

        '''
        - clean_score : (tf.Tensor) output of the neural network in thee clean run after the post-processing
        - labels : (tf.Tensor) label. Directly taken from the dataset.
        - num_classes : (int) number of classes in the dataset
        '''

    def __evaluate_clean_labels(self, x) -> tf.Tensor:
        """
        Function to transform the output of network in a form confrontable with the labels.

        output : (tf.Tensor) 
        """
        pass
        
    def clean_output(self) -> tuple:
        '''
        Compute the metrics related to the clean run.

        output : (tuple) of data that will be used to fill the report file. 
        
        Insert a null in the tuple for each metric that is not computed in the clean run.
        This is because the cardinality of the tuple must match the one of the faulty_output function.
        '''
        pass

    def faulty_output(self, faulty_scores) -> tuple:
        '''
        input : (tf.Tensor) output of the neural network in the faulty run after the post-processing
        Compute the metrics and return a tuple that will be used to fill the report file.
        '''
        
    
    def get_header(self):

        '''
        These are the names of the columns that will be used in the CSV file.
        The number of columns must match the number of metrics returned by the clean_output and faulty_output functions.
        '''

        return ("col1", 
                "col2",
                "col3",
                "col4",
                ...)
