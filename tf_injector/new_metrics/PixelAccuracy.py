from tf_injector.new_metrics.metric import Metric
import numpy as np
import tensorflow as tf

def compute_pixel_accuracy(x1, x2):
    diff = x1 == x2
    # totalPixels = x1.shape[0] * tf.math.reduce_prod( x1.shape[1:] )
    correctPixels = tf.math.reduce_sum( 
        tf.cast(diff, np.uint64),
        axis = [1,2]
    )
    return correctPixels

class PixelAccuracyMetric(Metric):
    def __init__(self, clean_scores, clean_labels, labels):
        # clean_scores = tf.argmax(clean_scores, axis=-1, output_type=tf.dtypes.uint16)
        super().__init__(None, clean_labels, labels)
       
    def clean_metric(self):
        return compute_pixel_accuracy(self.clean_labels, self.labels)
        
    def clean_output(self):
       return [1]

    def faulty_output(self, faulty_scores, with_respect_to_labels=False):
        # faulty_scores = tf.argmax(faulty_scores, axis=-1, output_type=tf.dtypes.uint16)
        reference = self.clean_labels
        if with_respect_to_labels:
            reference = self.labels
        return compute_pixel_accuracy(
            reference,
            faulty_scores,
        )
