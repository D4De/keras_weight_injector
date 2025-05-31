from tf_injector.new_metrics.metric import Metric
import numpy as np
import tensorflow as tf

def compute_pixel_accuracy(x1, x2):
    x1 = tf.cast(x1, tf.int64)
    x2 = tf.cast(x2, tf.int64)

    # x1 : (data, height, width)
    # x2 : (data, height, width)

    diff = x1 == x2
    # totalPixels = x1.shape[0] * tf.math.reduce_prod( x1.shape[1:] )
    correctPixels = tf.math.reduce_sum( 
        tf.cast(diff, np.uint64),
        axis = [0,1,2]
    ) # (data,)
    #correctPixels = tf.reduce_mean(correctPixels) # single value
    totalPixels = tf.math.reduce_sum(
        tf.cast(tf.ones_like(x1), np.uint64),
        axis = [0,1,2]
    ) 
    return correctPixels / totalPixels

class PixelAccuracy(Metric):
    def __init__(self, clean_scores, labels, num_classes):
        # clean_scores = tf.argmax(clean_scores, axis=-1, output_type=tf.dtypes.uint16)
        super().__init__(clean_scores, labels, num_classes)
        if clean_scores is not None:
            self.num_classes = clean_scores.shape[-1] 
            self.clean_labels = self.__evaluate_clean_labels(clean_scores) 

    def __evaluate_clean_labels(self, x):
        """
        clean_score : (batch, height, width, num_classes)
        clean_label : (batch, height, width)
        """
        x = tf.argmax(x, axis=-1)
        return x
        
    def clean_output(self):
        metric = compute_pixel_accuracy(self.clean_labels, self.labels)
        tupl = (float(metric.numpy()), None)
        return tupl

    def faulty_output(self, faulty_scores) -> tuple[float, float]:
        
        faulty_label = self.__evaluate_clean_labels(faulty_scores)
        output = list()

        # metric with (faulty_scores, labels)
        metric_on_labels = compute_pixel_accuracy(
            self.labels,
            faulty_label,
        )
        metric_on_labels = float(metric_on_labels.numpy())
        output.extend([metric_on_labels])

        # metric with (faulty_scores, clean_labels)
        metric_on_clean_score = compute_pixel_accuracy(
            self.clean_labels,
            faulty_label,
        )
        metric_on_clean_score = float(metric_on_clean_score.numpy())
        output.extend([metric_on_clean_score])

        return output
        
    
    def get_header(self):
        return ("pixel_accuracy_on_label", "pixel_accuracy_on_golden")
